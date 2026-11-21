"""Opt-in frozen generated programs in the existing ASAP/IPC/LangGraph runner.

The original baseline and the default explicit-action runner remain available.
This command replaces only the decision service; the same worker owns DFS and
every removal is checked by the same actual ASAP physics implementation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'worker'))
sys.path.insert(0, str(ROOT/'tools'))

from run_explicit_action_experiment import parser, execute
from run_integrated_repair import checked
from run_luna_experiment import read, write, sha
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.skill_registry import retrieve
from orchestrator.knowledge_store import resolve_bundle, verify_pin, read_json
from orchestrator.skill_usage import summarize_usage


from orchestrator.generated_execution_audit import (
    _unambiguous_decisions, execution_bindings, _actual_selector_bound, physical_usage_audit,
)


def resolve_knowledge_inputs(args):
    """Opt in once; never reread mutable ACTIVE during this run."""
    store = getattr(args, 'knowledge_store', None)
    pin_path = getattr(args, 'knowledge_pin', None)
    revision = getattr(args, 'knowledge_revision', None)
    if store and pin_path:
        raise ValueError('choose_knowledge_store_or_pin')
    if not store and not pin_path:
        if revision:
            raise ValueError('knowledge_revision_requires_store')
        if not args.skill_library:
            raise ValueError('skill_library_or_knowledge_bundle_required')
        return None
    if args.skill_library or args.memory:
        raise ValueError('knowledge_bundle_conflicts_with_explicit_memory_or_library')
    if pin_path:
        if revision:
            raise ValueError('pin_already_fixes_revision')
        pin = read_json(pin_path)[0]
        if args.knowledge_purpose == 'evaluation' and pin.get('purpose') != 'evaluation':
            raise ValueError('evaluation_requires_evaluation_run_pin')
        bundle = verify_pin(pin, allow_fixture=args.allow_fixture_library)
    else:
        bundle = resolve_bundle(store, revision, purpose=args.knowledge_purpose,
                                allow_fixture=args.allow_fixture_library)
        pin = bundle['pin']
    args.memory, args.skill_library = Path(bundle['memory']), Path(bundle['skills'])
    return pin


def main(argv=None):
    p = parser()
    p.description = __doc__
    p.add_argument('--skill-library', type=Path)
    p.add_argument('--lesson-library', type=Path)
    p.add_argument('--knowledge-policy', choices=('legacy_v1', 'unified_v1', 'agent_tools_v1'), default='legacy_v1')
    p.add_argument('--relational-knowledge', type=Path,
                   help='Frozen relational reference cards; requires agent_tools_v1.')
    p.add_argument('--knowledge-max-skills', type=int, default=16)
    p.add_argument('--knowledge-max-rounds', type=int, choices=(1, 2), default=2)
    p.add_argument('--knowledge-store', type=Path)
    p.add_argument('--knowledge-revision')
    p.add_argument('--knowledge-pin', type=Path)
    p.add_argument('--knowledge-purpose', choices=('development', 'evaluation'), default='development')
    p.add_argument('--allow-fixture-library', action='store_true')
    args = p.parse_args(argv)
    knowledge_pin = resolve_knowledge_inputs(args)
    if args.mode == 'llm' and args.ledger is None:
        raise ValueError('shared_approved_ledger_required_for_live_reuse')
    if args.mode == 'fixture' or args.fixture:
        raise ValueError('use_deterministic_for_declared_fixture_program_not_fixed_actions')
    if (args.provider_contract != 'explicit_action_provider_v5'
            or args.path_selection_policy != 'path_probe_round_v1'
            or args.fallback_policy != 'path_stage_v1'):
        raise ValueError('generated_programs_require_v5_and_current_path_guards')
    library = checked(args.skill_library, True)
    frozen_hash, snapshot = sha(library), read(library)
    lesson_path = checked(args.lesson_library, True) if args.lesson_library else None
    lessons = read(lesson_path) if lesson_path else None
    knowledge_memory = read(checked(args.memory, True)) if args.knowledge_policy == 'agent_tools_v1' and args.memory else None
    relational = None
    if getattr(args, 'relational_knowledge', None):
        if args.knowledge_policy != 'agent_tools_v1':
            raise ValueError('relational_knowledge_requires_agent_tools')
        from orchestrator.relational_knowledge import load_library
        relational = load_library(args.relational_knowledge)
    rows = retrieve(snapshot, allow_fixture=args.allow_fixture_library)
    if not rows:
        raise ValueError('no_admitted_generated_programs_in_frozen_library')
    if args.allow_fixture_library and args.mode != 'deterministic':
        raise ValueError('fixture_library_is_not_live_discovered_skill')
    services = []
    def factory(config, ledger, parameters):
        actor = ExplicitActionService(config, ledger, mode=parameters.mode,
            timeout_seconds=parameters.response_timeout, provider_contract=parameters.provider_contract,
            fallback_policy=parameters.fallback_policy)
        selector = SkillDiscoveryService(config, ledger) if parameters.mode == 'llm' else None
        service = GeneratedSkillService(snapshot, actor, selector, mode=parameters.mode,
            allow_fixture=parameters.allow_fixture_library, lesson_library=lessons,
            knowledge_policy=parameters.knowledge_policy, knowledge_max_skills=parameters.knowledge_max_skills,
            knowledge_memory=knowledge_memory, knowledge_max_rounds=parameters.knowledge_max_rounds,
            relational_knowledge=relational)
        services.append(service)
        return service
    code = execute(args, service_factory=factory, extension_metadata={
        'kind': 'generated_program_service_v1', 'library': str(library),
        'library_sha256': frozen_hash, 'fixture_program_library': args.allow_fixture_library,
        'generated_programs': len(rows), 'no_new_search_state_owner': True,
        'knowledge_pin': knowledge_pin, 'knowledge_policy': args.knowledge_policy,
        'knowledge_max_skills': args.knowledge_max_skills,
        'knowledge_max_rounds': args.knowledge_max_rounds,
        'relational_knowledge_sha256': relational['content_sha256'] if relational else None,
        'lesson_library_sha256': sha(lesson_path) if lesson_path else None})
    output = checked(args.run_dir, True)
    decisions = [item for service in services for item in service.decisions]
    selections = [item for service in services for item in service.selector_decisions]
    if args.knowledge_policy != 'legacy_v1':
        write(output/'knowledge_contexts.json', [v for s in services for v in s.knowledge_queries])
    write(output/'generated_skill_selections.json', selections)
    applied = [d for d in decisions if d['response']['metadata'].get('generated_program_executed')]
    report = read(output/'worker_result.json') if (output/'worker_result.json').is_file() else {}
    used_attempts, discarded = execution_bindings(decisions, report)
    physical_audit = physical_usage_audit(decisions, selections, report)
    audit = dict(library_sha256=frozen_hash, library_unchanged=sha(library)==frozen_hash,
        fixture_program_library=args.allow_fixture_library,
        generated_program_choices=len(applied), physical_attempts_of_generated_choices=len(used_attempts),
        generated_choices_replaced_before_physics=len(discarded),
        generated_choice_physical_passes=sum(row['physical_passes']
            for row in physical_audit['physical_usage_by_skill'].values()),
        actual_llm_skill_selections=sum(s['response'].get('origin')=='actual_llm'
            and s['response'].get('status')=='selected' for s in selections),
        selected_skill_ids=sorted({d['response']['metadata']['generated_skill_id'] for d in applied}),
        performance_benefit='NOT_RUN', automatic_skill_generation_not_part_of_this_execution=True,
        **physical_audit)
    write(output/'generated_skills_audit.json', audit)
    write(output/'skill_usage_summary.json', summarize_usage(decisions,
        report if report else None, selections, snapshot, allow_fixture=args.allow_fixture_library))
    print(audit, flush=True)
    return code or (0 if audit['library_unchanged'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
