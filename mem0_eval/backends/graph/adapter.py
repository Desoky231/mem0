from __future__ import annotations

import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from fastembed import TextEmbedding
from neo4j import GraphDatabase
from openai import OpenAI

from mem0_eval.backends.text.adapter import (
    TextMemoryAdapter,
    build_text_memory,
)


logger = logging.getLogger(__name__)
ACTIVE = "ACTIVE"
OBSOLETE = "OBSOLETE"


def _safe_cypher_identifier(value: Any, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", str(value)).strip("_").lower()
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        return fallback
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    return normalized


def _normal_name(value: Any) -> str:
    return " ".join(str(value).casefold().strip().split())


def _relation_family(relationship: str) -> str:
    aliases = {
        "current_city": "residence",
        "lives_at": "residence",
        "lives_in": "residence",
        "resides_at": "residence",
        "resides_in": "residence",
        "employed_by": "employment",
        "works_at": "employment",
        "works_for": "employment",
    }
    return aliases.get(relationship, relationship)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.casefold().startswith("json"):
            cleaned = cleaned[4:].strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class Entity:
    name: str
    entity_type: str


@dataclass(frozen=True)
class RelationTriple:
    source: str
    relationship: str
    destination: str


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


class PaperGraphExtractor:
    """Two-stage entity and relation extraction described in Mem0 section 2.2."""

    def __init__(self, client: OpenAI, *, model: str) -> None:
        self.client = client
        self.model = model

    def _json_completion(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_json_object(response.choices[0].message.content or "")

    def extract(
        self,
        *,
        messages: list[dict[str, str]],
        speaker: str,
        session_date: str,
        conversation_summary: str,
        recent_messages: list[str],
    ) -> tuple[list[Entity], list[RelationTriple]]:
        exchange = "\n".join(message["content"] for message in messages)
        context = (
            f"Key-knowledge summary:\n{conversation_summary or '(none)'}\n\n"
            "Previous messages:\n"
            + ("\n".join(recent_messages) if recent_messages else "(none)")
        )
        try:
            entity_result = self._json_completion(
                f"""Extract persistent entities from ONLY the NEW EXCHANGE.

The summary and previous messages are context only. Use them solely to resolve
pronouns, aliases, and omitted subjects. Never extract an entity or fact merely
because it appears in the context.

Target memory owner: {speaker}
Extract entities involved in persistent facts about the target memory owner.
The other speaker may resolve context but must not cause unrelated facts about
that other speaker to be stored in this owner scope.
Observation date: {session_date}

CONTEXT (reference resolution only):
{context}

NEW EXCHANGE (the only source of new graph knowledge):
{exchange}

Return JSON:
{{"entities": [{{"name": "canonical entity name", "type": "Person|Location|Event|Object|Concept|Date|Attribute|Organization"}}]}}
Do not invent entities.""",
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning(
                "Graph entity extraction failed for this exchange: %s", exc
            )
            return [], []
        entities = _entities_from_json(entity_result)
        if not entities:
            return [], []

        try:
            relation_result = self._json_completion(
                f"""Create directed relation triples from ONLY the NEW EXCHANGE.

The context may resolve pronouns and aliases, but it is not evidence. Every
triple must be supported by the new exchange. Prefer stable, descriptive
snake_case relation labels such as lives_in, works_at, prefers, owns, attended,
or happened_on. Include implicit relations only when they are unambiguous.

Target memory owner: {speaker}
Create relations specifically representing knowledge about the target memory
owner. Do not store unrelated facts belonging only to the other speaker.
Observation date: {session_date}
Entities: {json.dumps([entity.__dict__ for entity in entities])}

CONTEXT (reference resolution only):
{context}

NEW EXCHANGE (the only source of new graph knowledge):
{exchange}

Return JSON:
{{"relations": [{{"source": "entity", "relationship": "snake_case_label", "destination": "entity"}}]}}
Do not invent relations.""",
                max_tokens=1536,
            )
        except Exception as exc:
            logger.warning(
                "Graph relation extraction failed for this exchange: %s", exc
            )
            return entities, []
        return entities, _relations_from_json(relation_result)

    def query_entities(self, query: str) -> list[str]:
        try:
            result = self._json_completion(
                f"""Identify the key named or implied entities in this memory query.
Return JSON only as {{"entities": ["entity"]}}.
Query: {query}""",
                max_tokens=256,
            )
        except Exception as exc:
            logger.warning("Query entity extraction failed: %s", exc)
            return [query]
        values = result.get("entities", [])
        entities = [
            str(item).strip()
            for item in values
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
        return entities or [query]


def _entities_from_json(data: dict[str, Any]) -> list[Entity]:
    entities: list[Entity] = []
    seen: set[str] = set()
    for item in data.get("entities", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        key = _normal_name(name)
        if not key or key in seen:
            continue
        entity_type = _safe_cypher_identifier(
            item.get("type", "unknown"), fallback="unknown"
        )
        entities.append(Entity(name=name, entity_type=entity_type))
        seen.add(key)
    return entities


def _relations_from_json(data: dict[str, Any]) -> list[RelationTriple]:
    relations: list[RelationTriple] = []
    seen: set[tuple[str, str, str]] = set()
    for item in data.get("relations", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        destination = str(
            item.get("destination", item.get("target", ""))
        ).strip()
        relationship = _safe_cypher_identifier(
            item.get("relationship"), fallback="related_to"
        )
        key = (_normal_name(source), relationship, _normal_name(destination))
        if not source or not destination or key in seen:
            continue
        relations.append(
            RelationTriple(
                source=source,
                relationship=relationship,
                destination=destination,
            )
        )
        seen.add(key)
    return relations


class GraphUpdateResolver:
    """Resolve potentially conflicting active relations without deleting history."""

    FUNCTIONAL_RELATIONS = {
        "born_in",
        "born_on",
        "current_city",
        "employed_by",
        "has_age",
        "has_name",
        "lives_at",
        "lives_in",
        "located_at",
        "located_in",
        "married_to",
        "resides_at",
        "resides_in",
        "works_at",
        "works_for",
    }

    def __init__(self, client: OpenAI, *, model: str) -> None:
        self.client = client
        self.model = model

    def obsolete_relation_ids(
        self,
        *,
        new_relation: RelationTriple,
        candidates: list[dict[str, Any]],
        new_evidence: str,
    ) -> list[str]:
        different = [
            item
            for item in candidates
            if _normal_name(item.get("destination")) != _normal_name(
                new_relation.destination
            )
        ]
        if not different:
            return []
        if new_relation.relationship in self.FUNCTIONAL_RELATIONS:
            return [str(item["relation_id"]) for item in different]

        prompt = f"""Determine whether a new graph relation makes any existing
active relations obsolete. Invalidate only mutually exclusive or directly
contradicted facts. Multiple preferences, possessions, friends, activities,
and events normally coexist.

New evidence: {new_evidence}
New relation: {json.dumps(new_relation.__dict__)}
Existing active relations: {json.dumps(different, default=str)}

Return JSON only:
{{"obsolete_relation_ids": ["Neo4j relation id"], "reason": "short reason"}}
Use an empty list when the facts can coexist."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=384,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = _parse_json_object(
                response.choices[0].message.content or ""
            )
        except Exception as exc:
            logger.warning("Graph conflict resolver failed safely: %s", exc)
            return []
        valid_ids = {str(item["relation_id"]) for item in different}
        return [
            str(item)
            for item in parsed.get("obsolete_relation_ids", [])
            if str(item) in valid_ids
        ]


class TemporalNeo4jGraph:
    """Paper-specific graph behavior implemented directly over Neo4j."""

    def __init__(
        self,
        driver: Any,
        extractor: PaperGraphExtractor,
        resolver: GraphUpdateResolver,
        embedder: Embedder,
        *,
        top_k: int,
        threshold: float,
        entity_match_threshold: float = 0.95,
    ) -> None:
        self.driver = driver
        self.extractor = extractor
        self.resolver = resolver
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold
        self.entity_match_threshold = entity_match_threshold
        self._entity_keys: dict[tuple[str, str], str] = {}

    def add_exchange(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        speaker: str,
        session: str,
        session_date: str,
        conversation_summary: str,
        recent_messages: list[str],
        message_ids: list[str],
        source_dialogue_ids: list[str],
    ) -> dict[str, Any]:
        entities, relations = self.extractor.extract(
            messages=messages,
            speaker=speaker,
            session_date=session_date,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
        )
        entity_types = {
            _normal_name(entity.name): entity.entity_type for entity in entities
        }
        exchange_text = "\n".join(message["content"] for message in messages)
        stored: list[dict[str, Any]] = []
        for relation in relations:
            stored.append(
                self._store_relation(
                    relation,
                    entity_types=entity_types,
                    user_id=user_id,
                    session=session,
                    session_date=session_date,
                    message_ids=message_ids,
                    source_dialogue_ids=source_dialogue_ids,
                    evidence=exchange_text,
                )
            )
        return {
            "entities_extracted": len(entities),
            "relations_extracted": len(relations),
            "relations": stored,
        }

    def add_statement(self, statement: str, *, user_id: str) -> dict[str, Any]:
        now = _utc_now()
        return self.add_exchange(
            [{"role": "user", "content": statement}],
            user_id=user_id,
            speaker=user_id,
            session="direct",
            session_date=now,
            conversation_summary="",
            recent_messages=[],
            message_ids=[],
            source_dialogue_ids=[],
        )

    def _candidate_nodes(self, user_id: str) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (n:Mem0Entity {user_id: $user_id})
            RETURN n.entity_key AS entity_key, n.name AS name,
                   n.embedding AS embedding
            """,
            user_id=user_id,
        )
        return [dict(record) for record in records]

    def _resolve_entity(
        self,
        *,
        name: str,
        entity_type: str,
        user_id: str,
        session: str,
        session_date: str,
        creation_time: str,
        message_ids: list[str],
        source_dialogue_ids: list[str],
    ) -> str:
        key = _normal_name(name)
        embedding = self.embedder.embed(name)
        cache_key = (user_id, key)
        if cache_key in self._entity_keys:
            resolved_key = self._entity_keys[cache_key]
        else:
            candidates = self._candidate_nodes(user_id)
            exact = next(
                (
                    item
                    for item in candidates
                    if item.get("entity_key") == key
                ),
                None,
            )
            if exact is not None:
                resolved_key = str(exact["entity_key"])
            else:
                ranked = [
                    (
                        _cosine(
                            embedding, list(item.get("embedding") or [])
                        ),
                        item,
                    )
                    for item in candidates
                ]
                score, nearest = max(
                    ranked, default=(0.0, None), key=lambda item: item[0]
                )
                resolved_key = (
                    str(nearest["entity_key"])
                    if nearest is not None
                    and score >= self.entity_match_threshold
                    else key
                )
            self._entity_keys[cache_key] = resolved_key
        self.driver.execute_query(
            """
            MERGE (n:Mem0Entity {user_id: $user_id, entity_key: $entity_key})
            ON CREATE SET n.name = $name,
                          n.entity_type = $entity_type,
                          n.embedding = $embedding,
                          n.creation_time = $creation_time,
                          n.valid_from = $session_date,
                          n.valid_to = null,
                          n.status = $active
            SET n.observation_date = $session_date,
                n.session_id = $session,
                n.observation_dates = reduce(
                    acc = coalesce(n.observation_dates, []),
                    item IN [$session_date] |
                    CASE WHEN item IN acc THEN acc ELSE acc + item END
                ),
                n.session_ids = reduce(
                    acc = coalesce(n.session_ids, []), item IN [$session] |
                    CASE WHEN item IN acc THEN acc ELSE acc + item END
                ),
                n.message_ids = reduce(
                    acc = coalesce(n.message_ids, []), item IN $message_ids |
                    CASE WHEN item IN acc THEN acc ELSE acc + item END
                ),
                n.source_dialogue_ids = reduce(
                    acc = coalesce(n.source_dialogue_ids, []),
                    item IN $source_dialogue_ids |
                    CASE WHEN item IN acc THEN acc ELSE acc + item END
                )
            """,
            user_id=user_id,
            entity_key=resolved_key,
            name=name,
            entity_type=entity_type,
            embedding=embedding,
            creation_time=creation_time,
            session_date=session_date,
            session=session,
            message_ids=message_ids,
            source_dialogue_ids=source_dialogue_ids,
            active=ACTIVE,
        )
        return resolved_key

    def _active_candidates(
        self, *, user_id: str, source_key: str, relationship: str
    ) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (source:Mem0Entity {user_id: $user_id, entity_key: $source_key})
                  -[r:MEM0_RELATION]->(destination:Mem0Entity)
            WHERE r.relation_family = $relation_family AND r.status = $active
            RETURN elementId(r) AS relation_id,
                   destination.name AS destination,
                   destination.entity_key AS destination_key,
                   r.observation_date AS observation_date
            """,
            user_id=user_id,
            source_key=source_key,
            relation_family=_relation_family(relationship),
            active=ACTIVE,
        )
        return [dict(record) for record in records]

    def _store_relation(
        self,
        relation: RelationTriple,
        *,
        entity_types: dict[str, str],
        user_id: str,
        session: str,
        session_date: str,
        message_ids: list[str],
        source_dialogue_ids: list[str],
        evidence: str,
    ) -> dict[str, Any]:
        creation_time = _utc_now()
        source_key = self._resolve_entity(
            name=relation.source,
            entity_type=entity_types.get(_normal_name(relation.source), "unknown"),
            user_id=user_id,
            session=session,
            session_date=session_date,
            creation_time=creation_time,
            message_ids=message_ids,
            source_dialogue_ids=source_dialogue_ids,
        )
        destination_key = self._resolve_entity(
            name=relation.destination,
            entity_type=entity_types.get(
                _normal_name(relation.destination), "unknown"
            ),
            user_id=user_id,
            session=session,
            session_date=session_date,
            creation_time=creation_time,
            message_ids=message_ids,
            source_dialogue_ids=source_dialogue_ids,
        )
        candidates = self._active_candidates(
            user_id=user_id,
            source_key=source_key,
            relationship=relation.relationship,
        )
        exact = next(
            (
                item
                for item in candidates
                if item["destination_key"] == destination_key
            ),
            None,
        )
        if exact:
            self.driver.execute_query(
                """
                MATCH ()-[r:MEM0_RELATION]->()
                WHERE elementId(r) = $relation_id
                SET r.last_observation_date = $session_date,
                    r.last_session_id = $session,
                    r.observation_dates = reduce(
                        acc = coalesce(r.observation_dates, []),
                        item IN [$session_date] |
                        CASE WHEN item IN acc THEN acc ELSE acc + item END
                    ),
                    r.session_ids = reduce(
                        acc = coalesce(r.session_ids, []),
                        item IN [$session] |
                        CASE WHEN item IN acc THEN acc ELSE acc + item END
                    ),
                    r.message_ids = reduce(
                        acc = coalesce(r.message_ids, []), item IN $message_ids |
                        CASE WHEN item IN acc THEN acc ELSE acc + item END
                    ),
                    r.source_dialogue_ids = reduce(
                        acc = coalesce(r.source_dialogue_ids, []),
                        item IN $source_dialogue_ids |
                        CASE WHEN item IN acc THEN acc ELSE acc + item END
                    )
                """,
                relation_id=exact["relation_id"],
                session_date=session_date,
                session=session,
                message_ids=message_ids,
                source_dialogue_ids=source_dialogue_ids,
            )
            return {
                **relation.__dict__,
                "action": "reinforced",
                "relation_id": exact["relation_id"],
            }

        obsolete_ids = self.resolver.obsolete_relation_ids(
            new_relation=relation,
            candidates=candidates,
            new_evidence=evidence,
        )
        if obsolete_ids:
            self.driver.execute_query(
                """
                MATCH ()-[r:MEM0_RELATION]->()
                WHERE elementId(r) IN $relation_ids
                SET r.status = $obsolete, r.valid_to = $session_date
                """,
                relation_ids=obsolete_ids,
                obsolete=OBSOLETE,
                session_date=session_date,
            )
        triple_text = (
            f"{relation.source} {relation.relationship.replace('_', ' ')} "
            f"{relation.destination}"
        )
        records, _, _ = self.driver.execute_query(
            """
            MATCH (source:Mem0Entity {user_id: $user_id, entity_key: $source_key}),
                  (destination:Mem0Entity {
                      user_id: $user_id, entity_key: $destination_key
                  })
            CREATE (source)-[r:MEM0_RELATION]->(destination)
            SET r.relationship = $relationship,
                r.relation_family = $relation_family,
                r.user_id = $user_id,
                r.triplet_text = $triplet_text,
                r.triplet_embedding = $triplet_embedding,
                r.source_dialogue_ids = $source_dialogue_ids,
                r.message_ids = $message_ids,
                r.observation_date = $session_date,
                r.observation_dates = [$session_date],
                r.session_id = $session,
                r.session_ids = [$session],
                r.creation_time = $creation_time,
                r.valid_from = $session_date,
                r.valid_to = null,
                r.status = $active
            RETURN elementId(r) AS relation_id
            """,
            user_id=user_id,
            source_key=source_key,
            destination_key=destination_key,
            relationship=relation.relationship,
            relation_family=_relation_family(relation.relationship),
            triplet_text=triple_text,
            triplet_embedding=self.embedder.embed(triple_text),
            source_dialogue_ids=source_dialogue_ids,
            message_ids=message_ids,
            session_date=session_date,
            session=session,
            creation_time=creation_time,
            active=ACTIVE,
        )
        relation_id = str(records[0]["relation_id"]) if records else None
        return {
            **relation.__dict__,
            "action": "created",
            "relation_id": relation_id,
            "invalidated_relation_ids": obsolete_ids,
        }

    def _all_relations(self, user_id: str) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (source:Mem0Entity {user_id: $user_id})
                  -[r:MEM0_RELATION]->(
                      destination:Mem0Entity {user_id: $user_id}
                  )
            RETURN elementId(r) AS relation_id,
                   source.name AS source,
                   source.embedding AS source_embedding,
                   r.relationship AS relationship,
                   destination.name AS destination,
                   destination.embedding AS destination_embedding,
                   r.triplet_text AS triplet_text,
                   r.triplet_embedding AS triplet_embedding,
                   r.source_dialogue_ids AS source_dialogue_ids,
                   r.message_ids AS message_ids,
                   r.observation_date AS observation_date,
                   r.observation_dates AS observation_dates,
                   r.session_id AS session_id,
                   r.session_ids AS session_ids,
                   r.creation_time AS creation_time,
                   r.valid_from AS valid_from,
                   r.valid_to AS valid_to,
                   r.status AS status
            """,
            user_id=user_id,
        )
        return [dict(record) for record in records]

    def search(self, query: str, *, user_id: str) -> list[dict[str, Any]]:
        rows = self._all_relations(user_id)
        if not rows:
            return []
        query_embedding = self.embedder.embed(query)
        entity_embeddings = [
            self.embedder.embed(entity)
            for entity in self.extractor.query_entities(query)
        ]
        ranked: list[tuple[float, dict[str, Any], str]] = []
        for row in rows:
            source_embedding = list(row.pop("source_embedding") or [])
            destination_embedding = list(
                row.pop("destination_embedding") or []
            )
            triplet_embedding = list(row.pop("triplet_embedding") or [])
            entity_score = max(
                (
                    max(
                        _cosine(entity, source_embedding),
                        _cosine(entity, destination_embedding),
                    )
                    for entity in entity_embeddings
                ),
                default=0.0,
            )
            triplet_score = _cosine(query_embedding, triplet_embedding)
            score = max(entity_score, triplet_score)
            route = (
                "entity_centric"
                if entity_score >= triplet_score
                else "semantic_triplet"
            )
            if score >= self.threshold:
                ranked.append((score, row, route))
        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].get("status") == ACTIVE,
                str(item[1].get("observation_date") or ""),
            ),
            reverse=True,
        )
        return [
            {
                **row,
                "retrieval_score": round(score, 4),
                "retrieval_route": route,
            }
            for score, row, route in ranked[: self.top_k]
        ]

    def get_all(self, *, user_id: str) -> list[dict[str, Any]]:
        rows = self._all_relations(user_id)
        for row in rows:
            row.pop("source_embedding", None)
            row.pop("destination_embedding", None)
            row.pop("triplet_embedding", None)
        return rows

    def delete_all(self, *, user_id: str) -> None:
        self.driver.execute_query(
            """
            MATCH (n:Mem0Entity {user_id: $user_id})
            DETACH DELETE n
            """,
            user_id=user_id,
        )
        self._entity_keys = {
            key: value
            for key, value in self._entity_keys.items()
            if key[0] != user_id
        }

    def close(self) -> None:
        self.driver.close()


class GraphMemoryAdapter:
    """Composite Mem0 text memory plus paper-aligned temporal graph memory."""

    def __init__(
        self,
        text_memory: TextMemoryAdapter,
        graph_memory: TemporalNeo4jGraph,
    ) -> None:
        self.text_memory = text_memory
        self.graph_memory = graph_memory

    def add(self, statement: str, *, user_id: str) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                self.text_memory.add, statement, user_id=user_id
            )
            graph_future = executor.submit(
                self.graph_memory.add_statement, statement, user_id=user_id
            )
            return {
                "text": text_future.result(),
                "graph": graph_future.result(),
            }

    def add_exchange(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        speaker: str,
        session: str,
        session_date: str,
        conversation_summary: str,
        recent_messages: list[str],
        message_ids: list[str] | None = None,
        source_dialogue_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        ids = list(message_ids or source_dialogue_ids or [])
        dialogue_ids = list(source_dialogue_ids or ids)
        shared = {
            "user_id": user_id,
            "speaker": speaker,
            "session": session,
            "session_date": session_date,
            "conversation_summary": conversation_summary,
            "recent_messages": recent_messages,
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                self.text_memory.add_exchange, messages, **shared
            )
            graph_future = executor.submit(
                self.graph_memory.add_exchange,
                messages,
                **shared,
                message_ids=ids,
                source_dialogue_ids=dialogue_ids,
            )
            return {
                "text": text_future.result(),
                "graph": graph_future.result(),
            }

    def search(self, query: str, *, user_id: str) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                self.text_memory.search, query, user_id=user_id
            )
            graph_future = executor.submit(
                self.graph_memory.search, query, user_id=user_id
            )
            return {
                "memories": text_future.result(),
                "graph_memories": graph_future.result(),
            }

    def get_all(self, *, user_id: str) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                self.text_memory.get_all, user_id=user_id
            )
            graph_future = executor.submit(
                self.graph_memory.get_all, user_id=user_id
            )
            return {
                "memories": text_future.result(),
                "graph_memories": graph_future.result(),
            }

    def delete_all(self, *, user_id: str) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(
                self.text_memory.delete_all, user_id=user_id
            )
            graph_future = executor.submit(
                self.graph_memory.delete_all, user_id=user_id
            )
            text_result = text_future.result()
            graph_result = graph_future.result()
        return {
            "memories": text_result,
            "graph_memories": graph_result,
        }


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _neo4j_password() -> str:
    return os.getenv("NEO4J_LEGACY_PASSWORD") or _required_env(
        "NEO4J_PASSWORD"
    )


def verify_neo4j() -> dict[str, str]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(username, _neo4j_password()))
    try:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            "CALL dbms.components() YIELD versions "
            "RETURN versions[0] AS version LIMIT 1"
        )
        return {"uri": uri, "version": str(records[0]["version"])}
    finally:
        driver.close()


def build_graph_adapter(
    *,
    state_dir: Path,
    top_k: int = 10,
    threshold: float = 0.0,
) -> GraphMemoryAdapter:
    verify_neo4j()
    api_key = _required_env("DEEPSEEK_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    )
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), _neo4j_password()),
    )
    extractor = PaperGraphExtractor(client, model=model)
    graph = TemporalNeo4jGraph(
        driver,
        extractor,
        GraphUpdateResolver(client, model=model),
        FastEmbedder(),
        top_k=top_k,
        threshold=threshold,
    )
    text = build_text_memory(
        state_dir=state_dir / "text",
        top_k=top_k,
        threshold=threshold,
    )
    return GraphMemoryAdapter(text, graph)
