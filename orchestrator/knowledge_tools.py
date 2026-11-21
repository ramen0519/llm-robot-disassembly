"""Bounded, read-only knowledge tools over explicitly pinned snapshots.

No file paths, simulator, code execution, network or inferred success are tools.
Search is lexical; the calling model chooses queries and which documents to read.
"""
from copy import deepcopy
from itertools import zip_longest
import json
import re

from orchestrator.explicit_action_memory import FrozenActionMemory, MEMORY_BYTE_LIMIT
from orchestrator.knowledge_context import KnowledgeContext
from worker.explicit_action_contract import digest, validate_request

POLICY = "agent_tools_v1"
KINDS = ("all", "skill", "lesson", "memory")
TRIGGERS = ("all", "path_failed", "stability_failed", "plan_pose_guard_v1", "unknown")
MAX_TOOL_BYTES = 65536
OPERATION_FIELDS = ("tool", "query", "kind", "trigger", "offset", "limit", "document_id")
SEARCH_FIELDS = ("query", "kind", "trigger", "offset", "limit")
OPERATION_CONTRACT = "knowledge_operations_v2"


def operation_schema():
    """The same field rules constrain provider output and local dispatch."""
    search = dict(tool=dict(type="string", enum=["search"]),
        query=dict(type="string", maxLength=240), kind=dict(type="string", enum=list(KINDS)),
        trigger=dict(type="string", enum=list(TRIGGERS)),
        offset=dict(type="integer", minimum=0, maximum=100000),
        limit=dict(type="integer", minimum=1, maximum=8), document_id=dict(type="null"))
    variants = [search]
    # A typed read is useful metadata, not a search filter. Its prefix must
    # match the requested kind; catalog membership is checked at dispatch.
    for kind, prefixes in ((None, ("skill", "lesson", "memory", "current")),
                           ("skill", ("skill",)), ("lesson", ("lesson",)),
                           ("memory", ("memory", "current"))):
        read = {key: dict(type="null") for key in SEARCH_FIELDS}
        read.update(tool=dict(type="string", enum=["read"]),
            kind=dict(type="null") if kind is None else dict(type="string", enum=[kind]),
            document_id=dict(type="string", maxLength=256,
                pattern="^(?:" + "|".join(prefixes) + "):[^\\r\\n]+$"))
        variants.append(read)
    return dict(anyOf=[dict(type="object", additionalProperties=False,
        properties=fields, required=list(OPERATION_FIELDS)) for fields in variants])


def operation_instructions():
    schema = operation_schema()['anyOf']
    search = schema[0]['properties']
    kinds = ', '.join(search['kind']['enum'])
    triggers = ', '.join(search['trigger']['enum'])
    null_fields = ', '.join(k for k in SEARCH_FIELDS if k != 'kind')
    return (f"{OPERATION_CONTRACT}: Every operation includes exactly {', '.join(OPERATION_FIELDS)}. "
        f"search: query is a string (at most {search['query']['maxLength']} characters); "
        f"kind is one of {kinds}; trigger is one of {triggers}; offset is an integer "
        f"{search['offset']['minimum']}..{search['offset']['maximum']}; limit is an integer "
        f"{search['limit']['minimum']}..{search['limit']['maximum']}; document_id is null. "
        f"read: {null_fields} must all be null; document_id is an exact catalog ID; "
        "kind is null or the matching document kind (skill, lesson, memory). "
        "current: IDs have kind memory. No path or unknown ID can be read. "
        "Tool errors are not abstention. Correct failed operations within the remaining tool rounds; "
        "a later successful read of the same document_id or search with the same query resolves "
        "that failed lookup. Unrelated operations and an empty finish do not resolve it. "
        "finish=true cannot hide errors and does not increase the round or call allowance.")


def validate_operation(operation):
    if type(operation) is not dict or set(operation) != set(OPERATION_FIELDS):
        raise ValueError("invalid_knowledge_operation_fields")
    if operation['tool'] not in ('search', 'read'):
        raise ValueError('unknown_knowledge_tool')
    if operation['tool'] == 'read' and any(operation[k] is not None for k in SEARCH_FIELDS if k != 'kind'):
        raise ValueError('read_cannot_search')
    if operation['tool'] == 'search' and operation['document_id'] is not None:
        raise ValueError('search_cannot_read')
    types = {'string': str, 'integer': int, 'null': type(None)}
    def matches(value, spec):
        if type(value) is not types[spec['type']]:
            return False
        return (('enum' not in spec or value in spec['enum'])
            and ('maxLength' not in spec or len(value) <= spec['maxLength'])
            and ('minimum' not in spec or value >= spec['minimum'])
            and ('maximum' not in spec or value <= spec['maximum'])
            and ('pattern' not in spec or re.fullmatch(spec['pattern'], value) is not None))
    if not any(all(matches(operation[key], spec) for key, spec in branch['properties'].items())
               for branch in operation_schema()['anyOf']):
        raise ValueError('invalid_knowledge_' + operation['tool'])
    return operation


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def tokens(text):
    result = set(re.findall(r"[\w]+", text.casefold()))
    result.update(re.findall(r"[^\W_]+", text.casefold()))
    # Declared bilingual aliases, not semantic embeddings or hidden model calls.
    for term, alias in (("경로", "path"), ("안정성", "stability"), ("지지", "support"),
                        ("자세", "pose"), ("실패", "failed"), ("복구", "recovery")):
        if term in text:
            result.add(alias)
    return result


def trigger_of(path, stability):
    return "path_failed" if path is False else "stability_failed" if path is True and stability is False else "unknown"


class KnowledgeCatalog:
    def __init__(self, snapshot, lessons=None, memory=None, *, allow_fixture=False,
                 relational_knowledge=None):
        self.context = KnowledgeContext(snapshot, lessons, allow_fixture=allow_fixture)
        if memory is not None and encoded_size(memory) > MEMORY_BYTE_LIMIT:
            raise ValueError('knowledge_memory_byte_limit')
        self.memory = FrozenActionMemory(memory)
        self.fixture = bool(snapshot.get("fixture"))
        self.sources = dict(self.context.sources, memory_sha256=self.memory.snapshot["content_sha256"])
        self._documents = {}
        self.relational = None
        self.relational_documents = {}
        self.excluded_memory = []
        for identity, bundle in self.context._bundles.items():
            value = deepcopy(bundle)
            value["evidence_links"] = []  # Current-request linkage is attached by session().
            triggers = sorted({item["claim"]["trigger_kind"] for item in value["lessons"]}) or ["unknown"]
            self._add("skill:" + identity, "skill", value, triggers)
            for lesson in value["lessons"]:
                self._add("lesson:" + lesson["lesson_id"], "lesson", lesson,
                          [lesson["claim"]["trigger_kind"]])
        for record in self.memory.snapshot["records"]:
            result = record["result"]
            # Domain boundary, not a success/relevance filter. Unknown provenance
            # is not silently relabelled as measured RedMax evidence.
            origin = result["verifier_origin"]
            if origin.casefold() not in ('asap_redmax', 'actual_asap_redmax', 'actual_asap_artifacts_checked'):
                self.excluded_memory.append(dict(evidence_id=record["evidence_id"], reason="unrecognized_verifier_domain"))
                continue
            raw = result["raw"]
            upstream = raw.get('raw_upstream')
            value = dict(evidence_id=record["evidence_id"], source_run_id=record["source_run_id"],
                source_request_id=record["source_request_id"], source_state_id=record["source_state_id"],
                record_sha256=record["record_sha256"], scope=deepcopy(record["scope"]),
                source_choice=deepcopy(record["choice"]), parent_failure_evidence_id=record["parent_failure_evidence_id"],
                result={k: deepcopy(result[k]) for k in ("status", "path_success", "stability_success", "verifier_origin")},
                measured_endpoint=deepcopy(raw.get("endpoint_observation")),
                raw_stability_parts=deepcopy(upstream.get("parts_fall") if type(upstream) is dict else None),
                accepted_for_transition=None, deadline_exceeded=None,
                missing_acceptance_metadata=True, source_part_ids_are_target_ids=False,
                conditions_equivalent=False, physics_reuse_allowed=False, hard_pruning=False)
            from orchestrator.execution_observation import project_execution
            value.update(project_execution(result))
            value['missing_acceptance_metadata'] = any(value[k] is None for k in
                ('accepted_for_transition', 'deadline_exceeded'))
            if raw.get('stability_diagnostics') is not None:
                from worker.stability_diagnostics import project_stability_diagnostics
                value['stability_diagnostics'] = project_stability_diagnostics(raw['stability_diagnostics'])
            self._add("memory:" + record["evidence_id"], "memory", value,
                      [trigger_of(result["path_success"], result["stability_success"])])
        if relational_knowledge is not None:
            from orchestrator.relational_runtime import RelationalRuntime
            self.relational = RelationalRuntime(relational_knowledge)
            self.sources['relational_sha256'] = self.relational.fingerprint
            for card in self.relational.cards:
                kind = 'memory' if card['kind'] == 'experience' else card['kind']
                identity = kind + ':relational:' + digest(card['card_id'])
                self.relational_documents[card['card_id']] = identity
                # Reference-only even if the source calls itself registered. The
                # existing execution registry is the only action authority.
                self._documents[identity] = dict(document_id=identity, kind=kind,
                    content=dict(relational_card_id=card['card_id'], relational_summary=dict(
                        context={k: deepcopy((card.get('context') or {}).get(k)) for k in
                                 ('product_id', 'target_part_id', 'shape', 'topology', 'conditions')},
                        neighbors=deepcopy((card.get('context') or {}).get('neighbors', [])[:3]),
                        remaining_neighbors_require_read=True, status=card.get('status'),
                        evidence_ids=deepcopy(card.get('evidence_ids', [])[:8]))), triggers=['unknown'],
                    content_sha256=digest(card), reference_only=True,
                    evidence_domain=(card.get('context') or {}).get('conditions', {}).get('verifier_domain'),
                    target_verified=False, requires_fresh_physics=True)
        self.binding = digest(dict(sources=self.sources, documents=self._documents,
                                   excluded_memory=self.excluded_memory))

    def _add(self, identity, kind, content, triggers):
        doc = dict(document_id=identity, kind=kind, content=content, triggers=triggers,
                   content_sha256=digest(content), evidence_domain="ASAP_RedMax_part_level",
                   target_verified=False, requires_fresh_physics=True)
        if identity in self._documents and self._documents[identity] != doc:
            raise ValueError("knowledge_document_id_conflict")
        self._documents[identity] = doc

    def session(self, request):
        return KnowledgeSession(self, request)


class KnowledgeSession:
    def __init__(self, catalog, request):
        self.catalog = catalog
        self.request = validate_request(request)
        self.documents = deepcopy(catalog._documents)
        self.excluded_products = []
        if catalog.relational is not None:
            from orchestrator.relational_runtime import product_id
            current = product_id(request)
            self.excluded_products = [current] if current else []
            blocked = catalog.relational.excluded_ids(self.excluded_products)
            for identity in blocked:
                self.documents.pop(catalog.relational_documents[identity], None)
        # Current observations remain available even when no frozen memory was supplied.
        for origin in ("memory_references", "episode_feedback"):
            for record in self.request[origin]:
                identity = "current:" + record["evidence_id"]
                if identity in self.documents:
                    if self.documents[identity]['content']['record'] != record:
                        raise ValueError('conflicting_knowledge_evidence_id')
                    continue
                content = dict(record=deepcopy(record), origin=origin,
                    accepted_for_transition=record.get("accepted_for_transition"),
                    deadline_exceeded=record.get("deadline_exceeded"))
                self.documents[identity] = dict(document_id=identity, kind="memory", content=content,
                    content_sha256=digest(content), triggers=[trigger_of(record.get("path_success"), record.get("stability_success"))],
                    evidence_domain="current_request", target_verified=False, requires_fresh_physics=True)
        self.binding = digest(dict(catalog=catalog.binding, request=self.request))
        self.read_documents = {}
        self.events = []
        self.total_read_bytes = 0

    @staticmethod
    def summary(doc):
        content = doc["content"]
        if doc.get('reference_only'):
            return dict(document_id=doc['document_id'], kind=doc['kind'], triggers=doc['triggers'],
                        summary=deepcopy(content['relational_summary']),
                        content_sha256=doc['content_sha256'], reference_only=True,
                        relational_card_id=content['relational_card_id'], detail_bytes=None)
        descriptive = content.get("card", content.get("claim", content.get("result", content.get("record", {}))))
        if doc["kind"] == "memory":
            state = content.get("scope", {})
            choice = content.get("source_choice", content.get("record", {}).get("choice", {}))
            descriptive = dict(outcome=content.get("result", content.get("record", {})).get("status"),
                trigger=doc["triggers"], source_state_id=content.get("source_state_id"),
                source_part_id=choice.get('part_id'), direction=choice.get('direction_id'),
                supports=choice.get('support_ids'),
                remaining_count=len(state.get("remaining_parts", content.get("record", {}).get('remaining_parts', []))))
        return dict(document_id=doc["document_id"], kind=doc["kind"], triggers=doc["triggers"],
                    summary=json.dumps(descriptive, ensure_ascii=False)[:700],
                    content_sha256=doc["content_sha256"], detail_bytes=encoded_size(doc))

    def index(self):
        # This is a paged browse preview, not a relevance recommendation.
        previews = {kind: [self.summary(d) for d in sorted(self.documents.values(), key=lambda x: x["document_id"])
                    if d["kind"] == kind][:8] for kind in KINDS[1:]}
        result = dict(policy=POLICY, binding=self.binding, sources=self.catalog.sources,
            counts={kind: sum(d["kind"] == kind for d in self.documents.values()) for kind in KINDS[1:]},
            preview=previews, preview_order="document_id_not_recommendation", full_catalog_searchable=True,
            excluded_memory_count=len(self.catalog.excluded_memory),
            excluded_memory_preview=deepcopy(self.catalog.excluded_memory[:16]),
            tools=dict(search="Literal token overlap in names, descriptions, code, lessons and observed facts. Use kind/trigger filters or empty query to browse. offset paginates; no embeddings.",
                       read="Read one exact document_id from the pinned catalog. Never a path. Returns code, lessons and scoped facts; no execution."),
            operation_contract=operation_instructions(),
            current_evidence_ids=self.request["evidence_ids"],
            historical_ids_are_not_additional_action_citations=True,
            source_ids_are_not_target_ids=True, fixture=self.catalog.fixture)
        if self.catalog.relational is not None:
            result.update(relational_policy='relational_v1',
                relational_preview=self.search(query='', limit=6),
                reference_cards_do_not_authorize_execution=True)
            result['tools']['search'] += (' Relational cards also compare current candidate shape, contact '
                'relations and conditions. Query text and trigger refine evidence search; kind and offset '
                'remain model-selected. Prefix query with shape:, relations:, or situation: to choose '
                'one search channel; omit a prefix to combine all three. Missing features and condition '
                'differences are returned. Only confirmed training non-fixture cards are exposed; candidate '
                'skills remain references, never execution authorization.')
        return result

    def search(self, *, query, kind="all", trigger="all", offset=0, limit=6):
        validate_operation(dict(tool='search', query=query, kind=kind, trigger=trigger,
            offset=offset, limit=limit, document_id=None))
        wanted = tokens(query)
        matches = []
        for doc in self.documents.values():
            if doc.get('reference_only'):
                continue
            if kind != "all" and doc["kind"] != kind or trigger != "all" and trigger not in doc["triggers"]:
                continue
            matched = sorted(wanted & tokens(json.dumps(doc, ensure_ascii=False)))
            if wanted and not matched:
                continue
            matches.append((len(matched), doc, matched))
        matches.sort(key=lambda item: (-item[0], item[1]["document_id"]))
        result = dict(query=query, kind=kind, trigger=trigger, total_matches=len(matches), offset=offset,
            next_offset=offset+limit if offset+limit < len(matches) else None,
            results=[dict(self.summary(doc), matched_terms=terms) for _, doc, terms in matches[offset:offset+limit]],
            ranking="literal_token_overlap_then_id", relevance_is_not_validation=True)
        if self.catalog.relational is not None:
            found = self.catalog.relational.search(self.request, query=query, kind=kind,
                trigger=trigger, offset=offset, limit=limit, excluded_products=self.excluded_products)
            references = []
            for item in found['results']:
                identity = self.catalog.relational_documents[item['card_id']]
                if identity not in self.documents:
                    continue
                references.append(dict(self.summary(self.documents[identity]), retrieval=item))
            # Keep the legacy execution registry searchable independently of
            # reference cards; do not treat reference skills as executable.
            legacy = [row for row in result['results'] if not row.get('reference_only')]
            merged = []
            for pair in zip_longest(references, legacy):
                for item in pair:
                    if item is not None and len(merged) < limit:
                        merged.append(item)
            result.update(results=merged, ranking='relational_v1_with_legacy_registry',
                relational_audit=found['audit'], next_offset=found['next_offset'] or result['next_offset'],
                total_matches=found['total_matches'] + sum(not d.get('reference_only') for _,d,_ in matches))
        return result

    def read(self, document_id):
        if type(document_id) is not str or document_id not in self.documents:
            return dict(error="unknown_document_id", document_id=document_id)
        document = deepcopy(self.documents[document_id])
        if document.get('reference_only'):
            view = self.catalog.relational.read(document['content']['relational_card_id'],
                excluded_products=self.excluded_products, max_bytes=MAX_TOOL_BYTES-2048)
            if 'error' in view:
                return view
            document['content'] = view
        size = encoded_size(document)
        if size > MAX_TOOL_BYTES or (document_id not in self.read_documents and self.total_read_bytes + size > 98304):
            return dict(error="knowledge_read_byte_limit", document_id=document_id, bytes=size)
        if document_id not in self.read_documents:
            self.total_read_bytes += size
            self.read_documents[document_id] = deepcopy(document)
        return document

    def execute(self, operation):
        # Validate every field before dispatch. No getattr/eval/path interpretation.
        validate_operation(operation)
        if operation["tool"] == "search":
            result = self.search(**{k: operation[k] for k in ("query", "kind", "trigger", "offset", "limit")})
        else:
            document = self.documents.get(operation['document_id'])
            if document is not None and operation['kind'] not in (None, document['kind']):
                raise ValueError('read_kind_mismatch')
            result = self.read(operation["document_id"])
        self.events.append(dict(operation=deepcopy(operation), result=deepcopy(result)))
        return result

    def view(self, *, max_skills=16):
        skills = [deepcopy(d["content"]) for d in self.read_documents.values()
                  if d["kind"] == "skill" and not d.get('reference_only')]
        selected, excluded, size = [], [], 0
        for bundle in skills:
            n = encoded_size(bundle)
            if len(selected) >= max_skills or size+n > 98304:
                excluded.append(dict(skill_id=bundle["skill_id"], reason="planning_context_limit"))
                continue
            selected.append(bundle); size += n
        return dict(schema_version=1, kind="knowledge_context_v1", skills=selected,
            observations=[deepcopy(d) for d in self.read_documents.values()
                          if d["kind"] != "skill" or d.get('reference_only')],
            audit=dict(policy=POLICY, sources=deepcopy(self.catalog.sources), session_binding=self.binding,
                request_sha256=digest(self.request),
                observation_scope="explicitly_pinned_memory_and_current_request", retrieval="model_requested_search_and_read",
                available_skills=sum(d["kind"] == "skill" for d in self.documents.values()),
                selected_skill_ids=[b["skill_id"] for b in selected], excluded_skills=excluded,
                read_document_ids=list(self.read_documents), selected_skill_bytes=size,
                request_evidence_ids=self.request["evidence_ids"], new_citation_ids_added=False,
                hard_pruning=False, physics_reuse_allowed=False, performance_effect="UNVERIFIED"))
