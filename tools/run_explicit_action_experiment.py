"""Bounded real ASAP experiment: explicit actions through the final pair.

Fixture mode tests wiring using declared choices; it never represents an LLM.
All paid calls require a separately approved enabled config and explicit flag.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worker'))
sys.path.insert(0, str(ROOT / 'tools'))
from run_integrated_repair import checked, _excluded, ASAP, WSL, PYTHON
from run_luna_experiment import CONFIG_KEYS, read, write, sha, linux, validate_config


def validate_live_launch_config(config, service_factory, extension_metadata):
    """An explicitly budgeted Sol-only hierarchy may have an inactive scout.

    Ordinary Luna runs retain their positive-budget check. The factory still
    uses the shared transport gate, which refuses every zero-cap scout call.
    """
    metadata = extension_metadata or {}
    if config.get('max_calls') == 0 and service_factory is not None and metadata.get('kind') == 'luna_sol_hierarchy_v1':
        from orchestrator.hierarchy_budget import validate_policy
        policy = validate_policy(metadata['hierarchy_policy'])
        if not policy['enabled'] or config != policy['roles']['scout']:
            raise ValueError('inactive_scout_config_must_match_enabled_hierarchy')
        return
    validate_config(config, True)
from run_recorded import run_recorded
from worker.explicit_action_contract import DIRECTIONS, action_id, validate_request, validate_response


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('fixture', 'deterministic', 'llm'), default='deterministic')
    p.add_argument('--fixture', type=Path)
    p.add_argument('--memory', type=Path)
    p.add_argument('--memory-policy', choices=('legacy_v1', 'state_diverse_v1', 'state_balanced_v2', 'transfer_v1', 'assembly_balanced_v1'), default='legacy_v1')
    p.add_argument('--evaluation-manifest', type=Path)
    p.add_argument('--memory-max-records', type=int, default=4)
    p.add_argument('--provider-contract', choices=('explicit_action_provider_v2', 'explicit_action_provider_v3', 'explicit_action_provider_v4', 'explicit_action_provider_v5'),
                   default='explicit_action_provider_v2')
    p.add_argument('--fallback-policy', choices=('legacy_v1', 'path_stage_v1'), default='legacy_v1')
    p.add_argument('--path-selection-policy', choices=('legacy_v1', 'path_probe_round_v1'), default='legacy_v1')
    p.add_argument('--decision-failure-policy', choices=('legacy_v1', 'halt_v1'), default='legacy_v1')
    p.add_argument('--stability-diagnostics', action='store_true')
    p.add_argument('--physics-scene-policy', choices=('fresh_v1', 'path_scene_reuse_v1'),
                   default='fresh_v1', help='Opt-in path scene reuse; stability checks remain fresh.')
    p.add_argument('--llm-config', type=Path)
    p.add_argument('--allow-live-api', action='store_true')
    p.add_argument('--ledger', type=Path)
    p.add_argument('--assembly-dir', default=ASAP + '/assets/beam_assembly/original')
    p.add_argument('--base-part', default='6')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--budget', type=int, default=16)
    p.add_argument('--state-action-budget', type=int, default=12)
    p.add_argument('--max-gripper', type=int, default=3)
    p.add_argument('--max-pose', type=int, default=3)
    p.add_argument('--pose-catalog-policy', choices=('legacy_truncate_v1', 'complete_v1'),
                   default='legacy_truncate_v1')
    p.add_argument('--pose-transition-policy', choices=('legacy_v1', 'accepted_pose_carry_v1'),
                   default='legacy_v1', help='Retain a bound accepted parent transform as a fresh-verification candidate.')
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--wall-timeout', type=int, default=180)
    p.add_argument('--response-timeout', type=int, default=60)
    return p


class MailboxAdapter:
    def __init__(self, service=None, fixture=None):
        self.service, self.fixture = service, fixture
        self.decisions = []

    def rank(self, envelope):
        request = validate_request(envelope['features']['explicit_action'])
        outer = envelope.get('request', {})
        expected_outer = {k: request[k] for k in ('run_id', 'request_id', 'state_id', 'candidate_ids', 'base_part')}
        expected_outer['assembled_parts'] = request['remaining_parts']
        observed_outer = deepcopy(outer)
        for key in ('candidate_ids', 'assembled_parts'):
            if type(observed_outer.get(key)) is list:
                observed_outer[key] = sorted(observed_outer[key])
        if any(observed_outer.get(k) != v for k, v in expected_outer.items()):
            raise ValueError('outer_inner_request_mismatch')
        if self.fixture is None:
            self.service.timeout_seconds = min(self.service.timeout_seconds, envelope['timeout_seconds'])
            response = self.service.handle(request)
        else:
            response = self.fixture_response(request)
        self.decisions.append({'request': request, 'response': deepcopy(response)})
        return {'schema_version': 1, 'kind': 'rank_response',
            'request_id': request['request_id'], 'state_id': request['state_id'],
            'candidate_ids': response['candidate_order'], 'fallback_reason': response['fallback_reason'],
            'metadata': {'llm_live': response['metadata'].get('llm_live', False),
                         'explicit_action_response': response}}

    def fixture_response(self, request):
        for index, rule in enumerate(self.fixture['rules']):
            if sorted(rule['remaining_parts']) != request['remaining_parts']:
                continue
            matching = [p for p in request['pose_catalog'] if max(abs(a-b)
                for ar, br in zip(p['matrix'], rule['pose_matrix']) for a, b in zip(ar, br)) < 1e-6]
            if not matching:
                continue
            choice = {k: deepcopy(rule[k]) for k in ('part_id', 'skill_id', 'direction_id',
                      'second_direction_id', 'support_ids')}
            choice['pose_id'] = matching[0]['pose_id']
            choice['evidence_ids'] = ([r['evidence_id'] for r in request['memory_references']]
                                      if rule.get('cite_memory') else [])
            identity = action_id(request, choice)
            if identity in request['attempted_action_ids']:
                continue
            response = {'schema_version': 1, 'protocol': 'explicit_action_v1',
                'request_id': request['request_id'], 'state_id': request['state_id'],
                'candidate_order': [choice['part_id']] + [p for p in request['candidate_ids'] if p != choice['part_id']],
                'choice': choice, 'action_id': identity, 'fallback_reason': None,
                'metadata': {'selection_source': 'declared_fixture', 'llm_live': False,
                             'attempted': False, 'fixture_rule': index}}
            return validate_response(request, response)
        raise ValueError('no_matching_declared_fixture_action')

    def close(self):
        if self.service:
            self.service.close()


def audit_report(report):
    """Check the saved chain, exact selected conditions and terminal budget."""
    attempts, decisions = report['attempts'], report['decisions']
    errors = []
    by_request = {d['request']['request_id']: d for d in decisions}
    by_attempt = {a['event_id']: a for a in attempts}
    halted = [d for d in decisions if 'decision_halt_v1' in d['response'].get('metadata', {})]
    if halted:
        if (len(halted) != 1 or halted[-1] != decisions[-1]
                or report.get('decision_failure_policy') != 'halt_v1'
                or report['status'] != 'UNKNOWN' or report['success']):
            errors.append('invalid_decision_halt_report')
        for decision in halted:
            try:
                from worker.explicit_action_contract import validate_response as validate_halt
                validate_halt(decision['request'], decision['response'])
                reason = decision['response']['metadata']['decision_halt_v1']['reason']
                if report['stop_msg'] != 'decision halted: ' + reason:
                    errors.append('decision_halt_stop_mismatch')
            except (ValueError, TypeError, KeyError):
                errors.append('invalid_decision_halt_contract')
            if any(a['request_id'] == decision['request']['request_id'] for a in attempts):
                errors.append('decision_halt_dispatched_physics')
    if len(by_attempt) != len(attempts):
        errors.append('duplicate_attempt_identity')
    if report['success'] != (report['status'] == 'PASS'):
        errors.append('status_success_mismatch')
    if report['final_pair_calls'] != sum(len(a['remaining_parts']) == 2 for a in attempts):
        errors.append('terminal_count_mismatch')
    if len(by_request) != len(decisions) or report['total_n_eval'] != len(attempts):
        errors.append('identity_or_attempt_count')
    if len(attempts) > report['budget']:
        errors.append('budget_exceeded')
    for attempt in attempts:
        decision = by_request.get(attempt['request_id'])
        if not decision:
            errors.append('attempt_without_decision')
            continue
        validate_response(decision['request'], decision['response'])
        if decision['response']['choice'] != attempt['choice']:
            errors.append('choice_mismatch')
        if (attempt['remaining_parts'] != decision['request']['remaining_parts']
                or attempt['terminal_pair'] != (len(attempt['remaining_parts']) == 2)
                or attempt['action_id'] != decision['response']['action_id']):
            errors.append('attempt_scope_mismatch')
        if attempt['accepted_for_transition']:
            raw = attempt['raw']
            if (attempt['status'] != 'PASS' or raw.get('path_success') is not True
                    or raw.get('stability_success') is not True or attempt['deadline_exceeded']
                    or raw.get('chosen') != attempt['choice']):
                errors.append('invalid_pass_transition')
            choice = attempt['choice']
            selected_pose = next(p['matrix'] for p in decision['request']['pose_catalog'] if p['pose_id'] == choice['pose_id'])
            if (raw.get('pose') != selected_pose or raw.get('action') != DIRECTIONS[choice['direction_id']]
                    or sorted(raw.get('parts_fix', [])) != sorted(choice['support_ids'])
                    or raw.get('part_move', choice['part_id']) != choice['part_id']):
                errors.append('executed_conditions_mismatch')
            if any(raw.get(k) != 0 for k in ('hidden_direction_search_calls', 'hidden_pose_search_calls', 'hidden_support_search_calls')):
                errors.append('hidden_search')
    selected = report['selected_plan']
    if any(by_attempt.get(a.get('event_id')) != a for a in selected):
        errors.append('selected_action_not_observed_attempt')
    if (report['success'] and report.get('selected_plan_continuity_checked') is False
            or report.get('selected_plan_continuity_errors')):
        errors.append('planner_continuity_failure')
    if report['success']:
        remaining = list(selected[0]['remaining_parts']) if selected else []
        if decisions and remaining != decisions[0]['request']['remaining_parts']:
            errors.append('initial_state_omitted')
        if not selected:
            errors.append('empty_success_plan')
        for attempt in selected:
            if sorted(remaining) != sorted(attempt['remaining_parts']) or not attempt['accepted_for_transition']:
                errors.append('broken_selected_chain')
                break
            if attempt['choice']['part_id'] not in remaining:
                errors.append('selected_missing_part')
                break
            remaining.remove(attempt['choice']['part_id'])
        if len(remaining) != 1 or remaining != report['terminal_parts'] or not selected or not selected[-1]['terminal_pair']:
            errors.append('invalid_terminal_pair')
        if selected and len(selected) != len(selected[0]['remaining_parts']) - 1:
            errors.append('incomplete_plan')
        if selected:
            first = by_request[selected[0]['request_id']]['request']
            if first['base_part'] is not None and remaining != [first['base_part']]:
                errors.append('fixed_base_removed')
        for stats in (report.get('stats', {}), report.get('native_stats', {})):
            if stats and stats.get('sequence') != [a['choice']['part_id'] for a in selected]:
                errors.append('native_sequence_mismatch')
    elif selected:
        errors.append('unsuccessful_selected_plan')
    return {'validation_ok': not errors, 'errors': errors, 'plan_success': report['success'],
        'status': report['status'] if not errors else 'ERROR', 'verifier_calls': len(attempts),
        'final_pair_calls': report['final_pair_calls'],
        'pass_calls': sum(a['status'] == 'PASS' for a in attempts),
        'unknown_calls': sum(a['status'] == 'UNKNOWN' for a in attempts),
        'error_calls': sum(a['status'] == 'ERROR' for a in attempts),
        'path_checker_calls': sum(a['raw'].get('path_check_calls', 0) for a in attempts),
        'two_leg_controller_calls': sum(a['raw'].get('two_leg_controller_calls', 0) for a in attempts),
        'stability_checker_calls': sum(a['raw'].get('stability_check_calls', 0) for a in attempts),
        'sequence': [a['choice']['part_id'] for a in selected],
        'llm_choices': sum(d['response']['metadata'].get('llm_live') is True and d['response']['fallback_reason'] is None for d in decisions),
        'fallbacks': sum(d['response']['fallback_reason'] is not None and
            'decision_halt_v1' not in d['response']['metadata'] for d in decisions),
        'decision_halts': sum('decision_halt_v1' in d['response']['metadata'] for d in decisions),
        'memory_exposed_requests': sum(bool(d['request']['memory_references']) for d in decisions),
        # Historical name counts any citation, including this episode; retain it
        # for compatibility and report the provenance-specific counts separately.
        'memory_cited_requests': sum(bool(d['response']['choice'] and d['response']['choice']['evidence_ids']) for d in decisions),
        'frozen_memory_cited_requests': sum(bool(set(d['response']['choice']['evidence_ids']) &
            {r['evidence_id'] for r in d['request']['memory_references']}) for d in decisions if d['response']['choice']),
        'episode_cited_requests': sum(bool(set(d['response']['choice']['evidence_ids']) -
            {r['evidence_id'] for r in d['request']['memory_references']}) for d in decisions if d['response']['choice']),
        'actual_physics': report.get('actual_physics', False),
        'performance_benefit_demonstrated': False}


def execute(args, *, service_factory=None, extension_metadata=None):
    from orchestrator.explicit_action_service import ExplicitActionService
    from orchestrator.explicit_action_memory import append_action_observation, freeze_memory
    from orchestrator.file_service import FileRankServer
    from orchestrator.langgraph_workflow import Workflow
    from orchestrator.batch_budget import budget_summary
    scene_policy = getattr(args, 'physics_scene_policy', 'fresh_v1')
    if scene_policy not in ('fresh_v1', 'path_scene_reuse_v1'):
        raise ValueError('unknown_physics_scene_policy')
    pose_transition_policy = getattr(args, 'pose_transition_policy', 'legacy_v1')
    if pose_transition_policy not in ('legacy_v1', 'accepted_pose_carry_v1'):
        raise ValueError('unknown_pose_transition_policy')
    if ((getattr(args, 'provider_contract', 'explicit_action_provider_v2') in ('explicit_action_provider_v4', 'explicit_action_provider_v5'))
            != (getattr(args, 'path_selection_policy', 'legacy_v1') == 'path_probe_round_v1')):
        raise ValueError('provider_v4_requires_explicit_path_probe_policy')
    output = checked(args.run_dir, True)
    if output.exists():
        raise FileExistsError('new_run_required')
    assembly = str(_excluded(args.assembly_dir)).replace('\\', '/').rstrip('/')
    if getattr(args, 'evaluation_manifest', None):
        from worker.transfer_protocol import validate_manifest
        validate_manifest(checked(args.evaluation_manifest, True), assembly, args)
    elif assembly not in (ASAP + '/assets/beam_assembly/original',
                        '/opt/asap-research/prototype_completion_20261018/07009',
                        '/opt/asap-research/integrated_repair_20261018/02628',
                        '/opt/asap-research/assembly_01256_learning_20261018/01256'):
        raise ValueError('only_previously_authorized_official_assemblies')
    if (any(getattr(args, k) <= 0 for k in ('budget', 'state_action_budget', 'max_gripper', 'max_pose', 'timeout', 'wall_timeout', 'response_timeout'))
            or not 0 <= args.seed < 2**32 or args.response_timeout > 120):
        raise ValueError('invalid_budgets')
    if type(args.memory_max_records) is not int or not 0 <= args.memory_max_records <= 12:
        raise ValueError('invalid_memory_limit')
    base = None if args.base_part == 'none' else args.base_part
    if base is not None and (not base or any(c in base for c in '/\\')):
        raise ValueError('invalid_base_id')
    memory = checked(args.memory, True) if args.memory else None
    fixture_path = checked(args.fixture, True) if args.fixture else None
    if (args.mode == 'fixture') != bool(fixture_path):
        raise ValueError('fixture_mode_requires_declared_fixture_only')
    fixture = read(fixture_path) if fixture_path else None
    live = args.mode == 'llm'
    config = read(checked(args.llm_config)) if args.llm_config else {'enabled': False, 'max_calls': 0, 'max_usd': 0}
    if live != bool(args.allow_live_api) or not set(config).issubset(CONFIG_KEYS):
        raise ValueError('explicit_API_flag_and_documented_config_required')
    if live:
        if (set(config) != CONFIG_KEYS or config['enabled'] is not True or config['model'] != 'gpt-6-luna'
                or config['base_url'] != 'https://api.openai.com/v1' or config['api_key_env'] != 'OPENAI_API_KEY'):
            raise ValueError('complete_pinned_live_config_required')
    elif config.get('enabled') is not False or config.get('max_calls') != 0 or config.get('max_usd') != 0:
        raise ValueError('offline_requires_zero_API_budget')
    ledger = checked(args.ledger, True) if args.ledger else output / 'batch_ledger.sqlite'
    for suffix in ('-wal', '-shm', '-journal'):
        checked(Path(str(ledger)+suffix), True)
    output.mkdir(parents=True, exist_ok=False)
    knowledge_pin = (extension_metadata or {}).get('knowledge_pin')
    def check_knowledge_pin():
        if knowledge_pin is not None:
            from orchestrator.knowledge_store import verify_pin
            bound = verify_pin(knowledge_pin, allow_fixture=bool((extension_metadata or {}).get('fixture_program_library')))
            if args.memory is None or checked(args.memory, True) != Path(bound['memory']):
                raise ValueError('run_memory_does_not_match_knowledge_pin')
    check_knowledge_pin()
    if knowledge_pin is not None:
        write(output/'knowledge_pin.json', knowledge_pin)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    memory_hash = sha(memory) if memory else None
    files = sorted((ROOT/'worker').glob('*.py')) + sorted((ROOT/'orchestrator').glob('*.py'))
    files += [Path(__file__), ROOT/'tools/run_recorded.py', ROOT/'tools/run_luna_experiment.py', ROOT/'tools/run_integrated_repair.py']
    # Opt-in extensions supply a service, never a second search-state owner.
    if service_factory is not None:
        files += [ROOT/'tools/run_generated_skills.py']
        for name in (extension_metadata or {}).get('extra_source_files', []):
            extra = checked(ROOT/name)
            if not extra.is_relative_to(ROOT/'tools') or extra.suffix != '.py':
                raise ValueError('extension_source_must_be_workspace_tool')
            files.append(extra)
    hashes = {p.relative_to(ROOT).as_posix(): sha(checked(p)) for p in files}
    write(output/'source_before.json', hashes)
    for name in hashes:
        target = output/'implementation'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(ROOT/name), checked(target, True))
    run_id = 'explicit:' + hashlib.sha256(str(output).encode()).hexdigest()[:16] + ':' + args.mode
    write(output/'config.json', {'run_id': run_id, 'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'api_config': config, 'frozen_memory_sha256': memory_hash, 'actual_physics': True,
        'decision_backend': 'actual_LLM_if_response_valid' if live else args.mode,
        'performance_experiment': bool(getattr(args, 'evaluation_manifest', None)), 'extension': extension_metadata})
    if fixture is not None:
        write(output/'fixture.json', fixture)
    empty_budget = {'reserved_calls': 0, 'budget_committed_usd': '0', 'usage_cost_usd': '0'}
    write(output/'budget_before.json', budget_summary(ledger) if ledger.exists() else empty_budget)
    def emit(kind, **fields):
        with (output/'orchestrator_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(kind=kind, **fields),allow_nan=False)+'\n')
    def validate(state):
        check_knowledge_pin()
        if any(sha(ROOT/name) != h for name,h in hashes.items()) or (memory and sha(memory) != memory_hash):
            raise ValueError('source_or_memory_changed')
        return {'status': 'RUNNING'}
    def prepare(state):
        return {'artifacts': {'run_dir': str(output)}}
    def launch(state):
        if live:
            from local_api_key import key_environment
            key_context = key_environment()
        else:
            key_context = nullcontext('NOT_ACCESSED')
        with key_context:
            if live:
                validate_live_launch_config(config, service_factory, extension_metadata)
            if service_factory is None:
                service = ExplicitActionService(config,ledger,mode='llm' if live else 'deterministic',
                    timeout_seconds=args.response_timeout,
                    provider_contract=getattr(args, 'provider_contract', 'explicit_action_provider_v2'),
                    fallback_policy=getattr(args, 'fallback_policy', 'legacy_v1'))
            else:
                service = service_factory(config, ledger, args)
            adapter = MailboxAdapter(service, fixture)
            server = FileRankServer(output/'ipc',run_id,'pipeline',emit,rank_service=adapter)
            argv = WSL + ['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(args.wall_timeout)+'s',
                PYTHON,'-B','-u',linux(ROOT/'worker/run_explicit_actions.py'),'--asap-root',ASAP,
                '--assembly-dir',assembly,'--output-dir',linux(output),'--run-id',run_id]
            for key in ('seed','budget','state_action_budget','max_gripper','max_pose','timeout','response_timeout'):
                argv += ['--'+key.replace('_','-'),str(getattr(args,key))]
            if base is not None:
                argv += ['--base-part',base]
            if memory:
                argv += ['--frozen-memory',linux(memory)]
            argv += ['--memory-policy', args.memory_policy,
                     '--memory-max-records', str(args.memory_max_records),
                     '--fallback-policy', getattr(args, 'fallback_policy', 'legacy_v1'),
                     '--path-selection-policy', getattr(args, 'path_selection_policy', 'legacy_v1'),
                     '--decision-failure-policy', getattr(args, 'decision_failure_policy', 'legacy_v1'),
                     '--pose-catalog-policy', getattr(args, 'pose_catalog_policy', 'legacy_truncate_v1')]
            if getattr(args, 'stability_diagnostics', False):
                argv += ['--stability-diagnostics']
            if scene_policy != 'fresh_v1':
                argv += ['--physics-scene-policy', scene_policy]
            if pose_transition_policy != 'legacy_v1':
                argv += ['--pose-transition-policy', pose_transition_policy]
            if getattr(args, 'evaluation_manifest', None):
                argv += ['--evaluation-manifest', linux(args.evaluation_manifest)]
            errors = []
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    process = pool.submit(run_recorded,argv,ROOT,output/'execute')
                    while not process.done():
                        try:
                            server.serve_pending()
                        except Exception as error:
                            if not errors:
                                errors.append(type(error).__name__)
                                emit('service_error',error_type=type(error).__name__)
                        time.sleep(.05)
                    command = process.result()
            finally:
                write(output/'host_decisions.json',adapter.decisions)
                server.close()
            return {'status':'RUNNING' if command['exit_code']==0 and not errors else 'ERROR',
                    'metrics':{'process_exit':command['exit_code'],'seconds':command['elapsed_seconds'],'service_errors':errors}}
    def collect(state):
        report = read(output/'worker_result.json')
        audit = audit_report(report)
        write(output/'audit.json',audit)
        return {'status':audit['status'],'metrics':audit}
    def summarize(state):
        if not (output/'worker_result.json').is_file():
            return {'status':'ERROR','metrics':{'memory_export':'NOT_RUN_no_worker_result'}}
        report = read(output/'worker_result.json')
        from orchestrator.execution_observation import export_episode_memory
        records = export_episode_memory(report, output,
            (output/'planner/planner_report.json').relative_to(ROOT).as_posix())
        return {'metrics':{'memory_export':'PASS','memory_records':len(records)}}
    print(json.dumps({'phase':'start','run_id':run_id,'decision_mode':args.mode,'actual_physics':True,'API_enabled':live}),flush=True)
    with Workflow(output/'workflow.sqlite',dict(validate=validate,prepare=prepare,execute=launch,collect=collect,summarize=summarize),
                  lambda event:emit('graph_event',event=event)) as graph:
        result = graph.invoke({'run_id':run_id,'config':{'mode':args.mode,'source':hashes,'memory_hash':memory_hash}},run_id)
    write(output/'graph_state.json',result)
    same_source = all(sha(ROOT/name)==h for name,h in hashes.items())
    same_memory = memory is None or sha(memory)==memory_hash
    check_knowledge_pin()
    write(output/'host_preservation.json',{'source_unchanged':same_source,'memory_unchanged':same_memory})
    write(output/'budget_after.json',budget_summary(ledger) if ledger.exists() else empty_budget)
    print(json.dumps({'phase':'end','status':result['status'],'metrics':result['metrics'],
                      'source_unchanged':same_source,'memory_unchanged':same_memory}),flush=True)
    return 1 if result['status']=='ERROR' or not same_source or not same_memory else 0


if __name__ == '__main__':
    raise SystemExit(execute(parser().parse_args()))
