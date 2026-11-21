"""Pose catalog completeness policy; no geometry scoring or physics decisions."""
from copy import deepcopy
try:
    from .explicit_action_contract import MAX_POSE_CATALOG
except ImportError:
    from explicit_action_contract import MAX_POSE_CATALOG

POLICIES = ('legacy_truncate_v1', 'complete_v1')


class PoseCatalogError(ValueError):
    """A catalog bound, never evidence that a physical action is impossible."""
    def __init__(self, code, count, limit):
        super().__init__(code)
        self.audit = dict(code=code, generated_count=count, configured_limit=limit,
                          contract_limit=MAX_POSE_CATALOG, candidate_truncation=False,
                          physics_called=False, api_called=False)


def select_pose_catalog(unique, limit, policy='legacy_truncate_v1', evaluation=False):
    """Serialize every generated pose or reject overflow in opt-in complete mode.

    ``unique`` is the worker's existing matrix-hash dictionary. Hash ordering is
    only a stable serialization, never a physical recommendation. Retain legacy
    truncation exactly for saved experiments and default callers.
    """
    if policy not in POLICIES or type(limit) is not int or limit <= 0:
        raise ValueError('invalid_pose_catalog_policy_or_limit')
    if not unique:
        raise PoseCatalogError('empty_geometric_pose_catalog', 0, limit)
    if policy == 'complete_v1' or evaluation:
        if limit > MAX_POSE_CATALOG:
            raise PoseCatalogError('configured_pose_limit_exceeds_contract', len(unique), limit)
    if len(unique) > limit:
        if evaluation:
            raise PoseCatalogError('evaluation_pose_catalog_overflow_no_truncation', len(unique), limit)
        if policy == 'complete_v1':
            raise PoseCatalogError('complete_pose_catalog_overflow_no_truncation', len(unique), limit)
    return deepcopy([unique[h] for h in sorted(unique)[:limit]])
