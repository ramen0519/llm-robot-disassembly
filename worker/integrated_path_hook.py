"""Publish completed path attempts to both decision routes without owning DFS."""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps

from constraint_path_hook import ConstraintPathHook, write_new
from constraint_memory import FrozenConstraintMemory


class FeedbackClient:
    """One read-only snapshot at request publication, no replay or solver edits."""
    def __init__(self, client, hook):
        self.client, self.hook = client, hook

    @property
    def features(self):
        return self.client.features

    @features.setter
    def features(self, value):
        self.client.features = value

    def __getattr__(self, name):
        return getattr(self.client, name)

    def __call__(self, request):
        features = deepcopy(self.client.features)
        features['path_feedback'] = deepcopy(self.hook.completed_feedback)
        history = FrozenConstraintMemory(self.hook.records).retrieve(
            self.hook.session.context, request.assembled_parts, request.candidate_ids,
            max_records=4)['records']
        features['constraint_history'] = [dict(ref, origin='current_episode_history')
                                          for ref in history]
        self.client.features = features
        self.hook.emit('integrated_feedback_offered', request_id=request.request_id,
                       path_request_ids=[item['outcome']['request_id']
                                         for item in features['path_feedback']],
                       constraint_ids=[ref['record_id'] for ref in features['constraint_history']])
        return self.client(request)


class IntegratedPathHook(ConstraintPathHook):
    def __init__(self, session, signature):
        super().__init__(session, signature, controlled_probe=False)
        self.completed_feedback = []

    def _publish(self, outcome):
        if outcome['path_success'] is False:
            # Stability was never called for this failed path. Preserve null.
            outcome.update(stability_success=None, parts_fix=None, feasible=False)
        elif (type(outcome.get('stability_success')) is not bool
              or type(outcome.get('feasible')) is not bool):
            raise ValueError('path_feedback_before_full_verifier_completion')
        scope = outcome['scope']
        event_ids = {outcome['failed_evidence_id'], outcome['repaired_evidence_id']}
        records = [record for record in self.records
                   if record['evidence_event_id'] in event_ids]
        candidates = [part for part in scope['assembled_parts']
                      if part != self.session.context['task']['base_part']]
        references = FrozenConstraintMemory(records).retrieve(
            self.session.context, scope['assembled_parts'], candidates,
            max_records=4)['records']
        item = {'outcome': deepcopy(outcome), 'constraint_references': [
            dict(ref, origin='current_episode_history') for ref in references]}
        if len(self.completed_feedback) >= 1:
            raise ValueError('integrated_feedback_run_cap_exceeded')
        self.completed_feedback.append(item)
        self.emit('integrated_feedback_published', request_id=outcome['request_id'],
                  parent_call_id=outcome['parent_call_id'],
                  path_success=outcome['path_success'],
                  stability_success=outcome['stability_success'], feasible=outcome['feasible'],
                  constraint_ids=[ref['record_id'] for ref in item['constraint_references']])

    @contextmanager
    def installed(self, planner_class):
        client = self.session.client
        self.session.client = FeedbackClient(client, self)
        try:
            with super().installed(planner_class):
                original = planner_class._simulate

                @wraps(original)
                def simulate(*args, **kwargs):
                    before = len(self.outcomes)
                    raw = original(*args, **kwargs)
                    for outcome in self.outcomes[before:]:
                        self._publish(outcome)
                    return raw

                planner_class._simulate = simulate
                try:
                    yield self
                finally:
                    planner_class._simulate = original
        finally:
            self.session.client = client
            write_new(self.output / 'integrated_feedback.json', {
                'schema_version': 1, 'completed_path_feedback': self.completed_feedback,
                'controlled_choices': False, 'physics_result_reuse': False})
