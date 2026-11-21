"""Repeat only the known-Beam development check after code editing ended."""
from run_unseen_transfer import explicit,RUN
if __name__=='__main__':
    raise SystemExit(explicit('/opt/asap-research/unseen_transfer_pilot_20261019/development_beam',
        'B',RUN/'development/rule_final',RUN/'development_protocol.json'))
