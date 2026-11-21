"""Opt-in Luna/Sol hierarchy using the unchanged explicit-action ASAP worker."""
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tools"))
from run_explicit_action_experiment import parser as base_parser, execute
from run_integrated_repair import checked
from run_luna_experiment import read, write, sha
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.explicit_action_provider_v5 import restore_request_v5, encode_provider_decision_v5
from orchestrator.conditional_plan import match_pose, response_for
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService


class InjectedProvider:
    """Predeclared integration fixture. HTTP is not used. No model-quality claim."""
    def __init__(self, fixture):
        self.fixture, self.counts = deepcopy(fixture), {"scout": 0, "planner": 0}

    def transport(self, role):
        def call(url, headers, body, timeout):
            i = self.counts[role]; self.counts[role] += 1
            template = deepcopy(self.fixture[role][i])
            packet = json.loads(body["input"][0]["content"])
            if packet.get('projection_version') == 'strategy_shared_values_v1':
                from orchestrator.strategy_payload_projection import restore_context
                packet = restore_context(packet)
            if body.get('text', {}).get('format', {}).get('name') == 'knowledge_tool_requests':
                if 'knowledge_operations' not in template:
                    raise ValueError('explicit_knowledge_provider_fixture_required')
                value = dict(request_id=packet['request_id'], state_id=packet['state_id'],
                    operations=template['knowledge_operations'], finish=template.get('finish', False),
                    reason=template.get('reason', 'Explicit offline knowledge fixture'))
            elif packet.get('execution_contract') == 'strategy_plan_v1':
                request = restore_request_v5(packet['observation'])
                value = dict(request_id=request['request_id'], state_id=request['state_id'],
                    candidate_order=template.get('candidate_order', sorted(request['candidate_ids'])),
                    steps=template['steps'], recovery_skill_ids=[])
                if 'skill_review' in body.get('text', {}).get('format', {}).get('schema', {}).get('properties', {}):
                    # Tests must supply their review explicitly; never synthesize
                    # a model explanation from the locally chosen procedure.
                    if 'skill_review' in template:
                        value['skill_review'] = deepcopy(template['skill_review'])
            elif role == "planner":
                request = restore_request_v5(packet["observation"])
                if template.get("fixture_select_applicable_skill"):
                    card = next((c for c in packet["registered_skills"] if c.get("currently_proposed_choice")), None)
                    if card is not None:
                        choice = card["currently_proposed_choice"]
                        template["steps"] = [dict(expected_remaining=list(request["remaining_parts"]),
                            part_id=choice["part_id"], mode="registered_skill", pose_reference=None,
                            primitive=None, direction=None, second_direction=None, support_ids=[],
                            evidence_ids=[], registered_skill_id=card["skill_id"])]
                        template["recovery_skill_ids"] = [card["skill_id"]]
                value = dict(request_id=request["request_id"], state_id=request["state_id"],
                    candidate_order=sorted(request["candidate_ids"]), steps=template["steps"],
                    recovery_skill_ids=template.get("recovery_skill_ids", []))
                for step in value["steps"]:
                    if "orientation" in step:
                        step["pose_reference"] = match_pose(step.pop("orientation"), request)
                    elif "pose_catalog_index" in step:
                        step["pose_reference"] = request["pose_catalog"][step.pop("pose_catalog_index")]["pose_id"]
            else:
                request = restore_request_v5({k:packet[k] for k in
                    ("projection_version", "request", "shared_source_contexts")})
                choice = template["choice"]
                choice["pose_id"] = match_pose(template["orientation"], request)
                choice["evidence_ids"] = request["evidence_ids"][-1:] if template.get("cite") else []
                response = response_for(request, choice, sorted(request["candidate_ids"]), {})
                value = encode_provider_decision_v5(request, response)
            return dict(id="injected-"+role+"-"+str(i), model=body["model"], status="completed", error=None,
                output=[dict(type="message", role="assistant", status="completed",
                    content=[dict(type="output_text", text=json.dumps(value))])],
                usage=dict(input_tokens=100, output_tokens=100, total_tokens=200))
        return call


def parser():
    p = base_parser()
    from orchestrator.research_profile import PROFILE_ID
    from orchestrator.capacity_profile import PROFILE_ID as CAPACITY_PROFILE_ID
    p.add_argument('--research-profile', choices=(PROFILE_ID, CAPACITY_PROFILE_ID))
    p.add_argument('--release-certificate', type=Path)
    p.add_argument("--hierarchy-policy", type=Path, required=True)
    p.add_argument("--hierarchy-ledger", type=Path, required=True)
    p.add_argument("--planner-ledger", type=Path, required=True)
    p.add_argument("--skill-library", type=Path, required=True)
    p.add_argument("--provider-fixture", type=Path)
    p.add_argument("--allow-fixture-library", action="store_true")
    p.add_argument("--lesson-library", type=Path)
    p.add_argument("--knowledge-policy", choices=("legacy_v1", "unified_v1", "agent_tools_v1"), default="legacy_v1")
    p.add_argument('--relational-knowledge', type=Path,
                   help='Frozen relational reference cards; requires agent_tools_v1.')
    p.add_argument('--knowledge-runtime-feedback', action='store_true')
    p.add_argument("--knowledge-max-skills", type=int, default=16)
    p.add_argument("--knowledge-max-rounds", type=int, choices=(1, 2), default=2,
                   help="Extra metered planner requests per new plan; uses the same approved role/global caps.")
    p.add_argument("--lesson-runtime", action="store_true")
    p.add_argument("--max-skill-attempts", type=int, default=1)
    p.add_argument("--plan-provider-contract", choices=("conditional_plan_v1", "conditional_plan_v2", "conditional_plan_v3", "conditional_plan_v4", "strategy_plan_v1"),
                   default="conditional_plan_v1")
    p.add_argument("--recovery-support-policy", choices=("legacy_v1", "preserve_existing_v1"),
                   default="legacy_v1")
    p.add_argument("--planning-feedback-policy", choices=("legacy_v1", "path_evidence_v1", "repair_evidence_v2"),
                   default="legacy_v1")
    p.add_argument('--transport-failure-policy', choices=('legacy_v1', 'latch_on_error_v1'), default='legacy_v1')
    p.add_argument('--strategy-failure-policy', choices=('legacy_v1', 'model_ordered_failover_v1'),
                   default='legacy_v1')
    p.add_argument('--skill-review-policy', choices=('legacy_v1', 'skill_selection_review_v1'),
                   default='legacy_v1')
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    from orchestrator.research_profile import apply_episode_profile
    from orchestrator.capacity_profile import PROFILE_ID as CAPACITY_PROFILE_ID
    from orchestrator.capacity_profile import apply_episode_profile as apply_capacity_profile, validate_policy_binding
    effective_profile = (apply_capacity_profile(args, argv) if args.research_profile == CAPACITY_PROFILE_ID
                         else apply_episode_profile(args, argv))
    release = None
    if effective_profile and args.mode == 'llm':
        from orchestrator.research_release import validate_release_certificate
        if not args.release_certificate:
            raise ValueError('research_live_release_certificate_required')
        if effective_profile['profile_id'] == CAPACITY_PROFILE_ID:
            release = validate_release_certificate(args.release_certificate, root=ROOT,
                                                 profile_id=CAPACITY_PROFILE_ID)
        else:
            release = validate_release_certificate(args.release_certificate, root=ROOT)
    strategic = args.plan_provider_contract == 'strategy_plan_v1'
    if args.skill_review_policy != 'legacy_v1' and not strategic:
        raise ValueError('skill_review_requires_strategy_plan_contract')
    if args.strategy_failure_policy != 'legacy_v1' and not strategic:
        raise ValueError('strategy_failover_requires_strategy_plan_contract')
    if strategic and (args.decision_failure_policy != 'halt_v1'
            or args.recovery_support_policy != 'preserve_existing_v1'
            or args.planning_feedback_policy != 'repair_evidence_v2'
            or args.transport_failure_policy != 'latch_on_error_v1'):
        raise ValueError('strategy_requires_halt_preserve_supports_repair_feedback_and_transport_latch')
    policy_path = checked(args.hierarchy_policy)
    policy = validate_policy(read(policy_path))
    validate_policy_binding(policy, args.research_profile)
    live = args.mode == "llm"
    if (live != args.allow_live_api or live != policy["enabled"]
            or bool(args.provider_fixture) == live or args.mode == "fixture" or args.fixture
            or args.provider_contract != "explicit_action_provider_v5"
            or args.path_selection_policy != "path_probe_round_v1"):
        raise ValueError("explicit_hierarchy_mode_and_v5_path_round_required")
    if live and (not args.llm_config or read(checked(args.llm_config)) != policy["roles"]["scout"]
                 or args.allow_fixture_library):
        raise ValueError("scout_role_config_mismatch_or_fixture_live")
    library = checked(args.skill_library); snapshot = read(library); library_hash = sha(library)
    lesson_path = checked(args.lesson_library) if args.lesson_library else None
    lessons = read(lesson_path) if lesson_path else None
    lesson_hash = sha(lesson_path) if lesson_path else None
    knowledge_memory = read(checked(args.memory)) if args.knowledge_policy == 'agent_tools_v1' and args.memory else None
    relational = None
    if getattr(args, 'relational_knowledge', None):
        if args.knowledge_policy != 'agent_tools_v1':
            raise ValueError('relational_knowledge_requires_agent_tools')
        from orchestrator.relational_knowledge import load_library
        relational = load_library(args.relational_knowledge)
    budget = HierarchyBudget(checked(args.hierarchy_ledger, True), policy,
                             transport_failure_policy=args.transport_failure_policy)
    output = checked(args.run_dir, True)
    if output.exists():
        raise ValueError("new_episode_directory_required_no_partial_resume")
    injected = InjectedProvider(read(checked(args.provider_fixture))) if not live else None
    services = []
    def factory(config, ledger, parameters):
        scout_transport = budget.transport("scout", injected.transport("scout") if injected else None)
        planner_transport = budget.transport("planner", injected.transport("planner") if injected else None)
        if strategic:
            from orchestrator.strategy_planner import StrategyPlanService
            from orchestrator.strategy_service import StrategyDecisionService
            scout = StrategyPlanService(policy['roles']['scout'], ledger, scout_transport, role='scout',
                planning_feedback_policy=args.planning_feedback_policy,
                skill_review_policy=args.skill_review_policy,
                strategy_failure_policy=args.strategy_failure_policy)
            scout._live_transport = scout_transport.live
            planner = StrategyPlanService(policy['roles']['planner'], checked(args.planner_ledger, True),
                planner_transport, role='planner', planning_feedback_policy=args.planning_feedback_policy,
                skill_review_policy=args.skill_review_policy,
                strategy_failure_policy=args.strategy_failure_policy)
            service_class = StrategyDecisionService
        else:
            scout = ExplicitActionService(policy["roles"]["scout"], ledger, scout_transport,
                mode="llm", timeout_seconds=args.response_timeout, provider_contract=args.provider_contract,
                fallback_policy=args.fallback_policy)
            scout.service._live_transport = scout_transport.live
            planner = SolPlanService(policy["roles"]["planner"], checked(args.planner_ledger, True), planner_transport,
                provider_contract=args.plan_provider_contract, planning_feedback_policy=args.planning_feedback_policy)
            service_class = HierarchicalDecisionService
        planner._live_transport = planner_transport.live
        svc = service_class(scout, planner, snapshot, policy["episode"],
            timeout_seconds=args.response_timeout, allow_fixture=args.allow_fixture_library,
            journal_dir=output/"hierarchy_journal", recovery_support_policy=args.recovery_support_policy,
            planning_feedback_policy=args.planning_feedback_policy, lesson_library=lessons,
            lesson_runtime=args.lesson_runtime, max_skill_attempts=args.max_skill_attempts,
            knowledge_policy=args.knowledge_policy, knowledge_max_skills=args.knowledge_max_skills,
            knowledge_memory=knowledge_memory, knowledge_max_rounds=args.knowledge_max_rounds,
            knowledge_runtime_feedback=args.knowledge_runtime_feedback,
            relational_knowledge=relational,
            transport_guard=budget.transport_blocked if args.transport_failure_policy != 'legacy_v1' else None,
            **({'strategy_failure_policy': args.strategy_failure_policy,
                'skill_review_policy': args.skill_review_policy} if strategic else {}))
        services.append(svc)
        return svc
    # Injected mode cannot fall through to the network or expose a stored key.
    with (nullcontext() if live else patch.dict(os.environ, OPENAI_API_KEY="INJECTED_NOT_A_REAL_KEY")):
        with (nullcontext() if live else patch("urllib.request.OpenerDirector.open",
                                               side_effect=AssertionError("fixture_network_forbidden"))):
            code = execute(args, service_factory=factory, extension_metadata=dict(
                kind="luna_sol_hierarchy_v1", library_sha256=library_hash,
                fixture_program_library=args.allow_fixture_library, provider_fixture=not live,
                hierarchy_policy_sha256=sha(policy_path), hierarchy_policy=policy,
                plan_provider_contract=args.plan_provider_contract,
                decision_scope='strategy_plan_v1' if strategic else 'explicit_conditional_plan',
                decision_failure_policy=args.decision_failure_policy,
                recovery_support_policy=args.recovery_support_policy,
                planning_feedback_policy=args.planning_feedback_policy,
                transport_failure_policy=args.transport_failure_policy,
                strategy_failure_policy=args.strategy_failure_policy,
                skill_review_policy=args.skill_review_policy,
                pose_transition_policy=args.pose_transition_policy,
                lesson_library_sha256=lesson_hash, lesson_runtime=args.lesson_runtime,
                knowledge_policy=args.knowledge_policy, knowledge_max_skills=args.knowledge_max_skills,
                knowledge_runtime_feedback=args.knowledge_runtime_feedback,
                knowledge_max_rounds=args.knowledge_max_rounds,
                relational_knowledge_sha256=relational['content_sha256'] if relational else None,
                max_skill_attempts=args.max_skill_attempts,
                research_profile=effective_profile, release_certificate=release,
                # All orchestrator modules are already snapshotted by execute.
                extra_source_files=["tools/run_hierarchical_episode.py"],
                no_new_search_state_owner=True))
    decisions = [d for s in services for d in s.decisions]
    events = [e for s in services for e in s.events]
    if args.knowledge_policy != "legacy_v1":
        write(output/"knowledge_contexts.json", [v for s in services for v in s.knowledge_queries])
    plans = [d for s in services for d in s.planner_decisions]
    write(output/"hierarchy_decisions.json", decisions)
    write(output/"hierarchy_events.json", events)
    write(output/"hierarchy_plans.json", plans)
    skill_reviews = [e for e in events if e['kind'] in
        ('skill_selection_review_recorded', 'skill_selection_review_not_recorded')]
    write(output/'skill_selection_reviews.json', dict(
        policy=args.skill_review_policy, plan_reviews=skill_reviews,
        action_contexts=[dict(request_id=d['request']['request_id'],
            **deepcopy(d['response']['metadata']['skill_selection_review_context']))
            for d in decisions if 'skill_selection_review_context' in d['response']['metadata']],
        model_reason_is_self_report=True, selection_forced=False,
        physical_success_implied=False))
    report = read(output/"worker_result.json") if (output/"worker_result.json").exists() else {}
    bindings = {(d["request"]["request_id"], d["response"]["action_id"]):d for d in decisions}
    physical = []
    for a in report.get("attempts", []):
        d = bindings.get((a["request_id"], a["action_id"]))
        meta = d["response"]["metadata"] if d else {}
        physical.append(dict(request_id=a["request_id"], action_id=a["action_id"],
            role=meta.get("hierarchy_role", "unbound_worker_fallback"), status=a["status"],
            accepted=a.get("accepted_for_transition", False),
            backend=a["raw"].get("backend"), path_success=a["raw"].get("path_success"),
            stability_success=a["raw"].get("stability_success"),
            plan_id=meta.get("plan_id"), skill_id=meta.get("generated_skill_id")))
    roles = sorted({a["role"] for a in physical})
    audit = dict(actual_physics=report.get('actual_physics', False), provider_fixture=not live, automatic_http_retries=0,
        skill_review_policy=args.skill_review_policy,
        skill_selection_reviews_recorded=sum(e['kind']=='skill_selection_review_recorded' for e in skill_reviews),
        skill_selection_review_actions_without_new_model_choice=sum(
            d['response']['metadata'].get('skill_selection_review_context', {}).get('status')=='no_new_model_choice'
            for d in decisions),
        research_profile=effective_profile, release_certificate=release,
        lesson_library_unchanged=sha(lesson_path)==lesson_hash if lesson_path else True,
        lessons_supplied=sum(e["kind"] == "lessons_supplied" for e in events),
        status=report.get("status", "ERROR"), library_unchanged=sha(library)==library_hash,
        physical_attempts=len(physical), physics_bindings=physical,
        role_attempts={r:sum(a["role"]==r for a in physical) for r in roles},
        admitted_plans=sum(p["result"]["status"]=="planned" for p in plans),
        replanning_requests=sum(e["kind"]=="sol_requested" and not e["first_planning"] for e in events),
        generated_program_attempts=sum(bool(a["skill_id"]) for a in physical),
        generated_program_passes=sum(bool(a["skill_id"]) and a["accepted"] for a in physical),
        memory_visible_requests=sum(bool(d["request"]["memory_references"]) for d in decisions),
        memory_cited_requests=sum(bool(d["response"].get("choice") and
            set(d["response"]["choice"]["evidence_ids"]) &
            {m["evidence_id"] for m in d["request"]["memory_references"]}) for d in decisions),
        budget=budget.summary(), performance_benefit="NOT_RUN")
    write(output/"hierarchy_audit.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("physics_bindings","budget")}), flush=True)
    return code or (0 if audit["library_unchanged"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
