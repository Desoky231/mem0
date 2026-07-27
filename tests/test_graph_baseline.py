from __future__ import annotations

from types import SimpleNamespace

from mem0_eval.backends.graph.adapter import (
    ACTIVE,
    OBSOLETE,
    Entity,
    GraphMemoryAdapter,
    GraphUpdateResolver,
    PaperGraphExtractor,
    RelationTriple,
    TemporalNeo4jGraph,
    _relation_family,
    _safe_cypher_identifier,
)


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float("cairo" in text.casefold()), 1.0]


class FakeDriver:
    def __init__(self) -> None:
        self.calls = []

    def execute_query(self, query, **parameters):
        self.calls.append((query, parameters))
        if "RETURN n.entity_key" in query:
            return [], None, None
        if "RETURN elementId(r) AS relation_id" in query:
            return [{"relation_id": "new-edge"}], None, None
        return [], None, None


def test_generated_cypher_identifier_is_sanitized() -> None:
    assert (
        _safe_cypher_identifier(
            "platform/social_media", fallback="unknown"
        )
        == "platform_social_media"
    )
    assert (
        _safe_cypher_identifier("123 place", fallback="unknown")
        == "n_123_place"
    )
    assert _safe_cypher_identifier("/", fallback="unknown") == "unknown"


def test_relation_aliases_share_a_conflict_family() -> None:
    assert _relation_family("lives_in") == "residence"
    assert _relation_family("resides_in") == "residence"
    assert _relation_family("prefers") == "prefers"


def test_graph_extractor_uses_context_only_for_reference_resolution() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = []
            self.responses = [
                '{"entities":[{"name":"Ada","type":"Person"},'
                '{"name":"Cairo","type":"Location"}]}',
                '{"relations":[{"source":"Ada","relationship":"lives_in",'
                '"destination":"Cairo"}]}',
            ]

        def create(self, **arguments):
            self.calls.append(arguments)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=self.responses.pop(0)
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    extractor = PaperGraphExtractor(client, model="test-model")

    entities, relations = extractor.extract(
        messages=[{"role": "user", "content": "Ada: I live in Cairo."}],
        speaker="Ada",
        session_date="2 January 2024",
        conversation_summary="Ada previously lived in Giza.",
        recent_messages=["Ben: Where do you live?"],
    )

    assert entities == [
        Entity("Ada", "person"),
        Entity("Cairo", "location"),
    ]
    assert relations == [RelationTriple("Ada", "lives_in", "Cairo")]
    for call in completions.calls:
        prompt = call["messages"][0]["content"]
        assert "context" in prompt.casefold()
        assert "only source of new graph knowledge" in prompt.casefold()


def test_graph_extraction_failure_does_not_discard_text_pipeline() -> None:
    class FailingCompletions:
        def create(self, **arguments):
            raise RuntimeError("temporary model failure")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    extractor = PaperGraphExtractor(client, model="test-model")
    assert extractor.extract(
        messages=[{"role": "user", "content": "Ada: hello"}],
        speaker="Ada",
        session_date="2 January 2024",
        conversation_summary="",
        recent_messages=[],
    ) == ([], [])


def test_functional_relation_conflict_is_resolved_without_llm() -> None:
    client = SimpleNamespace(chat=None)
    resolver = GraphUpdateResolver(client, model="unused")
    obsolete = resolver.obsolete_relation_ids(
        new_relation=RelationTriple("Ada", "lives_in", "London"),
        candidates=[
            {
                "relation_id": "old-edge",
                "destination": "Cairo",
            }
        ],
        new_evidence="Ada: I moved to London.",
    )
    assert obsolete == ["old-edge"]


def test_nodes_receive_complete_temporal_provenance() -> None:
    driver = FakeDriver()
    graph = TemporalNeo4jGraph(
        driver,
        extractor=None,
        resolver=None,
        embedder=FakeEmbedder(),
        top_k=10,
        threshold=0.0,
    )

    graph._resolve_entity(
        name="Ada",
        entity_type="person",
        user_id="speaker-1",
        session="session_2",
        session_date="2 January 2024",
        creation_time="2024-01-02T00:00:00+00:00",
        message_ids=["D2:1", "D2:2"],
        source_dialogue_ids=["D2:1", "D2:2"],
    )

    query, parameters = next(
        item for item in driver.calls if "MERGE (n:Mem0Entity" in item[0]
    )
    for field in (
        "source_dialogue_ids",
        "message_ids",
        "observation_date",
        "session_id",
        "creation_time",
        "valid_from",
        "valid_to",
        "status",
    ):
        assert field in query
    assert parameters["message_ids"] == ["D2:1", "D2:2"]
    assert parameters["active"] == ACTIVE


def test_conflicting_edge_is_closed_and_new_edge_keeps_provenance() -> None:
    class Resolver:
        def obsolete_relation_ids(self, **arguments):
            assert arguments["candidates"][0]["relation_id"] == "old-edge"
            return ["old-edge"]

    class Graph(TemporalNeo4jGraph):
        def _resolve_entity(self, *, name, **context):
            return name.casefold()

        def _active_candidates(self, **context):
            return [
                {
                    "relation_id": "old-edge",
                    "destination": "Cairo",
                    "destination_key": "cairo",
                }
            ]

    driver = FakeDriver()
    graph = Graph(
        driver,
        extractor=None,
        resolver=Resolver(),
        embedder=FakeEmbedder(),
        top_k=10,
        threshold=0.0,
    )
    result = graph._store_relation(
        RelationTriple("Ada", "lives_in", "London"),
        entity_types={"ada": "person", "london": "location"},
        user_id="speaker-1",
        session="session_3",
        session_date="3 February 2024",
        message_ids=["D3:1"],
        source_dialogue_ids=["D3:1"],
        evidence="Ada: I moved to London.",
    )

    invalidation = next(
        item for item in driver.calls if "r.valid_to = $session_date" in item[0]
    )
    assert invalidation[1]["relation_ids"] == ["old-edge"]
    assert invalidation[1]["obsolete"] == OBSOLETE
    creation = next(
        item for item in driver.calls if "CREATE (source)-[r:MEM0_RELATION]" in item[0]
    )
    query, parameters = creation
    for field in (
        "source_dialogue_ids",
        "message_ids",
        "observation_date",
        "session_id",
        "creation_time",
        "valid_from",
        "valid_to",
        "status",
    ):
        assert field in query
    assert parameters["relation_family"] == "residence"
    assert result["invalidated_relation_ids"] == ["old-edge"]


def test_composite_adapter_returns_text_and_graph_results() -> None:
    class Store:
        def __init__(self, prefix):
            self.prefix = prefix
            self.context = None

        def add_exchange(self, messages, **context):
            self.context = context
            return {"store": self.prefix}

        def search(self, query, *, user_id):
            return [{"store": self.prefix, "query": query}]

        def get_all(self, *, user_id):
            return [{"store": self.prefix}]

        def delete_all(self, *, user_id):
            return None

    text = Store("text")
    graph = Store("graph")
    adapter = GraphMemoryAdapter(text, graph)
    added = adapter.add_exchange(
        [{"role": "user", "content": "Ada: hello"}],
        user_id="speaker-1",
        speaker="Ada",
        session="session_1",
        session_date="1 January 2024",
        conversation_summary="",
        recent_messages=[],
        message_ids=["D1:1"],
        source_dialogue_ids=["D1:1"],
    )
    retrieval = adapter.search("Where?", user_id="speaker-1")

    assert added == {
        "text": {"store": "text"},
        "graph": {"store": "graph"},
    }
    assert graph.context["message_ids"] == ["D1:1"]
    assert retrieval["memories"][0]["store"] == "text"
    assert retrieval["graph_memories"][0]["store"] == "graph"
