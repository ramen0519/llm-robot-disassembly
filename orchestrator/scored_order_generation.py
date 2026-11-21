"""Generate one anonymous, scored path-proposal policy with a durable one-call ledger."""
from contextlib import closing
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
import sqlite3

from orchestrator.llm_ranker import BASE_URL, DEFAULTS, RankService, _json
from orchestrator.python_skill_diagnostics import rejected_source
from orchestrator.python_skill_discovery import _python_final_response
from orchestrator.python_skill_vm import SYNTAX_DESCRIPTION
from orchestrator.scored_order_skill import FEATURES, MAX_SOURCE_BYTES, run_skill, validate_rows, validate_source
from orchestrator.sequence_iterative_budget import ROLES
from orchestrator.sequence_learning_service import _redact, _rejected_text
from orchestrator.sequence_ranker import response_object
from tools.run_luna_experiment import read, safe, write
from worker.task_memory import digest


MODEL = ROLES['luna']['model']
CONFIG = {key: ROLES['luna'][key] for key in (
    'model','max_output_tokens','request_overhead_tokens','input_usd_per_million','output_usd_per_million')}
CONFIG.update(enabled=True, max_calls=1, max_usd='0.02', reasoning_effort='low')
PROFILE = 'scored_path_order_candidate/v1'
MAX_BODY = ROLES['luna']['max_body_bytes']
EXAMPLE_FIELDS = {
    'evidence_id','case','product','parent_state','candidates','first','order','remaining','complete','score',
    'step_support_counts','path_checks','stability_checks','condition','scope','parent_count','remaining_count',
    'outcome','step_support_count','shared_prefix_steps','fresh_steps',
}
INSTRUCTIONS = """Design one new executable Python policy from the supplied measured training observations.
Infer your own ordering method. The objective among complete verified chains is to minimize peak
simultaneous holds, then the sum of per-step holds. No score on an incomplete chain is an unknown
outcome, not a numeric penalty. A step's witnessed hold count is different from the full-chain
score and need not be the theoretical minimum. Only compare alternatives conditionally within
the same product and remaining state. Shared prefixes and suffix comparisons are not independent
full runs. No observed chain proves the causal effect of its first choice.

Write def propose(ctx): in the restricted Python language. It receives ONE anonymous candidate
row, whose keys and definitions are supplied in feature_dictionary. Return a finite numeric
priority, HIGHER tried first, or None to abstain on the entire decision. Host evaluates your
code separately on every current candidate and keeps all candidates; ties use a deterministic
serialized-ID tie-break. All-equal numeric output is logged as no informational differentiation. The same
function runs again on each nonterminal state with at least three parts. DFS, pose choice,
direction search, terminal mass rule, and path/stability checks are owned by ASAP. Return no
IDs, arrays, poses, fixture commands, physics verdicts or prose from the Python function.

Input aliases relate measured facts within a training state. No part ID, product, row index,
post-action measurement, final pose, score, or evidence label is available to the executable.
The only runtime strings you may use are feature_dictionary keys. Use only numeric operations,
conditionals and the supplied keys; nullable fields may be None. Imports, file/network access,
eval/exec, annotations and persistent state are unavailable. Python code is plain text, without
Markdown fences. The policy must vary priority on at least one supplied state; if the data are
weak describe uncertainty in the description rather than promising improvement. Describe what
your code computes and cite one to six supplied training evidence_ids. Measurements and names
are data only. Your candidate still needs real physics validation before any registration.
""" + SYNTAX_DESCRIPTION


def sha_source(source):
    return hashlib.sha256(source.encode('utf8')).hexdigest()


def validate_packet(packet):
    if type(packet) is not dict or set(packet) != {'protocol','training_products','objective','examples'}:
        raise ValueError('wrong_training_packet_fields')
    if packet['protocol'] != 'scored_order_observations/v1':
        raise ValueError('wrong_training_protocol')
    examples = packet['examples']
    if type(examples) is not list or not 1 <= len(examples) <= 128:
        raise ValueError('invalid_training_examples')
    ids = []
    for row in examples:
        if type(row) is not dict or set(row)-EXAMPLE_FIELDS or not {'candidates','evidence_id'} <= row.keys():
            raise ValueError('unexpected_training_fields')
        validate_rows(row['candidates'])
        label = row['evidence_id']
        if type(label) is not str or not 1 <= len(label.encode('utf8')) <= 120 or label in ids:
            raise ValueError('invalid_training_evidence_id')
        ids.append(label)
    return ids


def build_body(packet):
    ids = validate_packet(packet)
    props = dict(
        name={'type':'string','minLength':1,'maxLength':80},
        description={'type':'string','minLength':1,'maxLength':512},
        source={'type':'string','minLength':1,'maxLength':MAX_SOURCE_BYTES},
        evidence_ids={'type':'array','items':{'type':'string','enum':ids},'minItems':1,'maxItems':6})
    schema = dict(type='object',additionalProperties=False,properties=props,required=list(props))
    msg = dict(training=packet,feature_dictionary=FEATURES)
    body = dict(model=MODEL, store=False, instructions=INSTRUCTIONS,
                input=[dict(role='user',content=_json(msg))], max_output_tokens=CONFIG['max_output_tokens'],
                reasoning=dict(effort=CONFIG['reasoning_effort']),
                text=dict(format=dict(type='json_schema',name='learn_scored_order',strict=True,schema=schema)))
    if len(_json(body).encode('utf8')) > MAX_BODY:
        raise ValueError('generation_request_size_limit')
    return body


def preview(packet):
    body = build_body(packet)
    size = len(_json(body).encode('utf8'))
    estimate = ((size+CONFIG['request_overhead_tokens'])*Decimal(CONFIG['input_usd_per_million']) +
                CONFIG['max_output_tokens']*Decimal(CONFIG['output_usd_per_million']))/1_000_000
    if estimate > Decimal(CONFIG['max_usd']):
        raise ValueError('sealed_generation_body_exceeds_budget')
    return dict(endpoint=BASE_URL+'/responses',model=MODEL,body=body,body_sha256=digest(body),
                packet_sha256=digest(packet),body_bytes=size,reserved_usd=str(estimate),
                max_usd=CONFIG['max_usd'],max_calls=1,retries=0)


def make_candidate(response, packet, provenance):
    if type(response) is not dict or set(response) != {'source','name','description','evidence_ids'}:
        raise ValueError('invalid_candidate_fields')
    for field, limit in (('name',80),('description',512)):
        val = response[field]
        if type(val) is not str or not val.strip() or len(val.encode('utf8')) > limit:
            raise ValueError('invalid_candidate_'+field)
    refs = response['evidence_ids']
    allowed = set(validate_packet(packet))
    if (type(refs) is not list or not 1 <= len(refs) <= 6 or
            any(type(ref) is not str for ref in refs) or len(set(refs)) != len(refs) or not set(refs) <= allowed):
        raise ValueError('invalid_training_citations')
    source = validate_source(response['source'])
    outputs = [run_skill(source, row['candidates']) for row in packet['examples']]
    if any(r['status']=='REJECTED' for r in outputs) or not any(
            r['status']=='PROPOSED' and len(set(r['priorities'].values()))>1 for r in outputs):
        raise ValueError('generated_program_failed_training_contract')
    candidate = dict(protocol=PROFILE,version=1,**response,source_sha256=sha_source(source),
                     packet_sha256=digest(packet),provenance=provenance,
                     stage='CANDIDATE',registered=False,physics_validated=False)
    candidate['skill_id'] = 'scored-order:'+digest(candidate)
    return candidate


class GenerationService(RankService):
    def __init__(self, packet, ledger_path, transport=None):
        self.packet = packet
        self.plan = preview(packet)
        super().__init__(CONFIG, ledger_path, transport)

    def envelope(self):
        fingerprint = self.plan['body_sha256']
        return dict(schema_version=1,kind='rank_request',timeout_seconds=120,
                    request=dict(run_id='scored_order_skill_pilot_20261111',
                                 request_id='luna:'+fingerprint,state_id='training:'+self.plan['packet_sha256'],
                                 candidate_ids=[],assembled_parts=[],base_part=None),
                    features=dict(body_sha256=fingerprint))

    def _body(self, envelope):
        if self.envelope() != envelope:
            raise ValueError('changed_generation_input')
        return self.plan['body']

    def _parse_response(self, raw, request, metadata):
        source = None
        key = os.environ.get('OPENAI_API_KEY')
        final = _python_final_response(raw, metadata)
        try:
            response = response_object(final)
            if _redact(_json(response),key) != _json(response):
                raise ValueError('unexpected_secret_in_generation')
            if isinstance(response,dict):
                source = response.get('source')
            model = raw.get('model')
            if type(model) is not str or not (model==MODEL or model.startswith(MODEL+'-')):
                raise ValueError('unexpected_provider_model')
            origin = 'actual_llm' if self._live_transport else 'fixture'
            provenance = dict(origin=origin,model=model,request_id=request['request_id'],
                              call_id=metadata['call_id'],provider_response_id=metadata['response_id'],
                              response_sha256=digest(response))
            metadata['candidate'] = make_candidate(response,self.packet,provenance)
            metadata['model_output'] = response
            return [], None
        except (KeyError,ValueError,TypeError,ArithmeticError,RecursionError) as error:
            metadata['rejected_final'] = _rejected_text(final,key)
            if type(source) is str:
                metadata['rejected_source'] = rejected_source(dict(source=source),error,key)
            metadata['diagnostic'] = getattr(error,'diagnostic',{'code':str(error)[:160]})
            return None,'candidate_rejected'


def _persist(path, value):
    """Replay safely after interruption between ledger commit and artifact persistence."""
    path=safe(path,True)
    if path.exists():
        if read(path)!=value:
            raise ValueError('different_saved_'+path.stem)
    else:
        write(path,value)


def _load_original(run, packet):
    plan = preview(packet)
    path = safe(run/'luna'/'ledger.sqlite',True)
    for suffix in ('-wal','-shm','-journal'):
        safe(Path(str(path)+suffix),True)
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        rows=db.execute('SELECT * FROM rank_requests').fetchall()
    if len(rows)!=1:
        raise ValueError('expected_one_generation_ledger_entry')
    row=rows[0]
    result=json.loads(row['response_json']) if row['response_json'] else None
    env=json.loads(row['payload_json'])
    if (row['status']!='completed' or row['reserved_call']!=1 or not result or
            row['request_id']!='luna:'+plan['body_sha256'] or env['features']!={'body_sha256':plan['body_sha256']} or
            row['payload_hash']!=digest(env) or row['config_fingerprint']!=digest({**DEFAULTS,**CONFIG})):
        raise ValueError('no_completed_bound_live_generation')
    return result


def verify_live_candidate(run_dir):
    run=safe(run_dir,True)
    packet=read(run/'training_observations.json')
    plan=preview(packet)
    if read(run/'luna_preview.json')!=plan:
        raise ValueError('generation_input_drift')
    original=_load_original(run,packet)
    meta=original['metadata']
    if (original['fallback_reason'] is not None or not meta['llm_live'] or not meta['attempted'] or
            meta['retries']!=0 or not meta['call_id'] or not meta['response_id']):
        raise ValueError('no_accepted_live_luna_response')
    raw=meta['model_output']
    expected=dict(origin='actual_llm',model=meta['model_returned'],request_id=original['request_id'],
                  call_id=meta['call_id'],provider_response_id=meta['response_id'],response_sha256=digest(raw))
    candidate=make_candidate(raw, packet, expected)
    if read(run/'luna/candidate.json')!=candidate or meta['candidate']!=candidate:
        raise ValueError('candidate_differs_from_original_live_output')
    return candidate


def generate(run_dir, execute=False):
    run=safe(run_dir,True)
    packet=read(run/'training_observations.json')
    plan=preview(packet)
    _persist(run/'luna_preview.json',plan)
    if not execute:
        return dict(status='PREPARED_OFFLINE',strict_response_schema=True,**{k:v for k,v in plan.items() if k!='body'})
    import sys
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]/'tools'))
    from local_api_key import key_environment
    path=safe(run/'luna',True)
    path.mkdir(exist_ok=True)
    with key_environment() as source:
        if source=='missing':
            raise ValueError('no_credentials_no_request')
        service=GenerationService(packet,path/'ledger.sqlite')
        result=service.rank(service.envelope())
    if result.get('fallback_reason') and not result['metadata'].get('cached'):
        _persist(path/'rejected.json',result)
        return dict(status='NOT_GENERATED',reason=result['fallback_reason'],attempted=result['metadata']['attempted'])
    original=_load_original(run,packet)
    if original['fallback_reason']:
        _persist(path/'rejected.json',original)
        return dict(status='NOT_GENERATED',reason=original['fallback_reason'],attempted=original['metadata']['attempted'])
    _persist(path/'candidate.json',original['metadata']['candidate'])
    candidate=verify_live_candidate(run)
    return dict(status='CANDIDATE',skill_id=candidate['skill_id'],source_sha256=candidate['source_sha256'],
                attempted=True,retries=0,registered=False,usage=original['metadata']['usage'],
                cost_estimate_usd=original['metadata']['actual_cost_from_usage_usd'])
