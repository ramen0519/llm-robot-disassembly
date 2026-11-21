"""Retrieve frozen skills/evidence, select, execute, and log the proposal.

The returned contract remains the unchanged worker candidate permutation.
Selection of a registered callable does not advance the physical/search state.
"""
from copy import deepcopy

from .llm_ranker import RankService
from .skill_selector import SkillSelector, PROTOCOL, response_fixture
from .skill_library import FrozenSkills, SkillMemory, execute_skill, digest, canonical


class SkillPipeline:
    def __init__(self, config, ledger, frozen_directory, context, memory_store,
                 selector_mode='disabled', transport=None):
        if selector_mode not in ('disabled', 'scripted_test', 'live'):
            raise ValueError('unsupported_selector_mode')
        if selector_mode != 'live' and config.get('enabled') is not False:
            raise ValueError('offline_modes_require_disabled_config')
        if selector_mode == 'live' and config.get('enabled') is not True:
            raise ValueError('live_mode_requires_enabled_config')
        self.skills = FrozenSkills(frozen_directory)
        self.context = deepcopy(context)
        self.store = memory_store
        self.mode = selector_mode
        self.selector = SkillSelector(config, ledger, transport)

    def rank(self, envelope: dict) -> dict:
        # Validate before logging/querying memory, and isolate caller data.
        try:
            RankService._validate_envelope(envelope)
        except (ValueError, TypeError, RecursionError):
            return self.selector._response(envelope, 'invalid_request', self.selector._metadata())
        request = deepcopy(envelope['request'])
        rid = request['request_id']
        binding = digest(dict(envelope=envelope, library=self.skills.library_sha256,
                              memory=self.skills.memory.content_sha256, mode=self.mode))
        previous = self.store.get('decision_response', rid)
        if previous is not None:
            if previous['binding'] != binding:
                return self.selector._response(envelope, 'request_id_conflict', self.selector._metadata())
            response = deepcopy(previous['response'])
            response['metadata'].update(cached=True, attempted=False, llm_live=False, invocation_reused=True)
            return response
        cards, evidence = self.skills.retrieve(self.context, request)
        offered = {c['skill_id']: c for c in cards}
        enriched = deepcopy(envelope)
        enriched['request'].update(offered_skill_ids=list(offered), library_sha256=self.skills.library_sha256,
                                   decision_protocol=PROTOCOL)
        enriched['features']['registered_skills'] = cards
        enriched['features']['evidence_memory'] = evidence
        enriched['features']['scope'] = 'Exact-context bounded observations, not physical facts for untested actions.'
        enriched['timeout_seconds'] = min(30., float(envelope['timeout_seconds']) - 5.)
        if enriched['timeout_seconds'] <= 0:
            enriched['timeout_seconds'] = float(envelope['timeout_seconds']) / 2
        self.store.put('retrieval', rid, dict(request=request, binding=binding,
            offered_skill_ids=list(offered), evidence=evidence, library_sha256=self.skills.library_sha256,
            memory_sha256=self.skills.memory.content_sha256))
        if self.mode == 'scripted_test':
            # Tests exercise the same strict parser and executor; no inference is claimed.
            chosen = next((sid for sid in offered if sid != 'native_order_v1'), 'native_order_v1')
            selection, reason = self.selector._parse(response_fixture(enriched['request'], chosen), enriched['request'])
            response = self.selector._response(enriched, reason, self.selector._metadata(), selection)
            response['metadata']['offline_fixture'] = True
        else:
            response = self.selector.rank(enriched)
        metadata = response['metadata']
        reason = response['fallback_reason']
        sid = metadata.get('selected_skill_id') if reason is None else 'native_order_v1'
        skill_error = None
        try:
            if sid not in offered:
                raise ValueError('unoffered_skill')
            order = execute_skill(offered[sid], self.context, request, evidence)
        except Exception as error:
            skill_error = type(error).__name__
            sid = 'native_order_v1'
            order = execute_skill(offered[sid], self.context, request, evidence)
        source = ('scripted_test' if self.mode == 'scripted_test' else 'llm') if not reason and not skill_error else 'deterministic_fallback'
        invocation = dict(invocation_id=rid, request=request, binding=binding, skill_id=sid,
            callable_id=offered[sid]['callable_id'], code_sha256=offered[sid]['code_sha256'],
            selected_by=source, selection_fallback=reason, execution_error_type=skill_error,
            applied_order=order, source_evidence_ids=offered[sid]['source_evidence_ids'],
            library_sha256=self.skills.library_sha256, physical_verdict=None,
            state_mutations=False, internal_verifier_calls=0)
        self.store.put('invocation', rid, invocation)
        metadata.update(skill_id=sid, selected_by=source, ranking_source=source,
            llm_fallback_reason=reason, skill_error_type=skill_error,
            skill_code_sha256=offered[sid]['code_sha256'], skill_invocation_id=rid,
            library_sha256=self.skills.library_sha256, retrieved_evidence_ids=evidence['evidence_ids'],
            skill_evidence_ids=offered[sid]['source_evidence_ids'],
            skill_applied=True, candidate_pruning=False, physics_bypass=False)
        response.update(candidate_ids=order, fallback_reason=None)
        self.store.put('decision_response', rid, dict(binding=binding, response=deepcopy(response)))
        return response
