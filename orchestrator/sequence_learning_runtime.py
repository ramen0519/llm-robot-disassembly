"""No-model candidate validation adapter; never used as a free LLM policy.

Only the order is proposed here. The original ASAP worker owns all physics.
Every VM input is saved so registration can replay exactly the same contract.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from worker.task_memory import digest
from orchestrator.file_service import canonical
from orchestrator.sequence_learning import execute_sequence_candidate


class CandidateValidationService:
    def __init__(self, service, candidate):
        if service.config.get('enabled') is not False:
            raise ValueError('candidate_validation_requires_disabled_api')
        self.service, self.candidate = service, deepcopy(candidate)
        self.calls = []

    def rank(self, envelope):
        metadata = dict(llm_live=False, attempted=False, retries=0,
                        validation_role='sequence_candidate',
                        candidate_sha256=self.candidate['content_sha256'],
                        program_sha256=self.candidate['program_sha256'])
        try:
            enriched = self.service.prepare(envelope)
            request = enriched['request']
            context = enriched['features']['sequence_knowledge']
            metadata['context_sha256'] = digest(context)
            result = execute_sequence_candidate(self.candidate, request['candidate_ids'], context)
            metadata['candidate_execution'] = result
            metadata['knowledge_audit'] = enriched['features']['knowledge_audit']
            reason = None if result['status'] == 'PROPOSED' else 'candidate_' + result['status'].lower()
            response = self.service._response(envelope, reason, metadata, result.get('candidate_order'))
            self.calls.append(dict(request=deepcopy(request), context=deepcopy(context),
                                   envelope_sha256=hashlib.sha256(canonical(envelope)).hexdigest(),
                                   response=deepcopy(response)))
            return response
        except (ValueError, KeyError, TypeError) as error:
            metadata['contract_error'] = type(error).__name__
            return self.service._response(envelope, 'candidate_input_rejected', metadata)

    def trace(self, response_paths, read, sha):
        """Match persisted IPC outputs, not assumed filenames or synthetic PASS."""
        responses = {}
        for path in response_paths:
            response = read(path)
            identity = response['request_id']
            if identity in responses:
                raise ValueError('duplicate_validation_response')
            responses[identity] = (path, response)
        calls = []
        for item in self.calls:
            path, response = responses[item['request']['request_id']]
            # FileRankServer appends only these two transport provenance fields.
            expected = deepcopy(item['response'])
            expected['metadata'].update(backend='pipeline', envelope_sha256=item['envelope_sha256'])
            if response != expected:
                raise ValueError('saved_candidate_response_differs')
            calls.append(dict(request=item['request'], context=item['context'],
                              response_ref={'path':str(path), 'sha256':sha(path)}))
        return dict(protocol='sequence_candidate_trace_v1',
                    candidate_sha256=self.candidate['content_sha256'],
                    program_sha256=self.candidate['program_sha256'], calls=calls)
