"""One command: scout/plan/physics -> freeze -> Python learning -> next episode.

LangGraph sequences immutable episodes. It never resumes an interrupted solver,
edits a generated candidate, or treats a program's output as a physics verdict.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tools"))
from run_integrated_repair import checked
from run_luna_experiment import read, write, sha
from run_recorded import run_recorded
from run_automatic_learning_cycle import validate_plan, task_argv, runtime_hashes
from prepare_episode_learning import prepare_episode_learning
from orchestrator.learning_cycle_workflow import LearningCycleWorkflow
from orchestrator.explicit_action_memory import freeze_cumulative_memory
from orchestrator.skill_library_merge import freeze_skill_union
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy
from worker.task_memory import digest


def research_learning_packet(manifest, run_id):
    """Check all recorded clocks before spending on lessons or Python generation."""
    from run_python_skill_cycle import python_generation_packet, attach_debug_inputs
    from run_skill_discovery import source_case
    from tools.python_source_budget import observation_binding
    prepared = [source_case(case) for case in manifest['cases']]
    bindings = {item['spec']['case_id']: observation_binding(item) for item in prepared}
    packet = attach_debug_inputs(python_generation_packet(manifest, run_id), prepared,
                                 run_id, source_observation=True)
    return packet, bindings


def assert_cycle_sources(hashes):
    if any(sha(ROOT/name) != value for name, value in hashes.items()):
        raise ValueError('research_cycle_source_changed_before_command')


def candidate_admitted(result, exit_code):
    """A claimed admission cannot override a failed child process."""
    return exit_code == 0 and result.get('admission', {}).get('admitted') is True


def target_skip_reason(learning, research_profile):
    if research_profile and (not learning.get('new_ids') or learning.get('learning_error')
                             or learning.get('merge_conflict')):
        return learning.get('learning_error') or learning.get('merge_conflict') or 'no_new_published_skill'
    return None


def candidate_infrastructure_error(result):
    """A verifier exception is not physical feedback for another paid revision."""
    if result.get('source_unchanged') is False:
        return 'learning_source_changed'
    if result.get('status') == 'ERROR' or any(
            case.get('outcome') == 'ERROR' for case in result.get('cases', [])):
        return 'learning_case_infrastructure_error'
    return None


def candidate_followup_args(folder, result, *, live, ledger):
    """Next allocated candidate, never a transport retry or extra allocation."""
    if not live:
        return []
    if result.get('status') in ('NOT_REGISTERED', 'PREFLIGHT_REJECTED'):
        return ['--refine-from', folder]
    if result.get('status') == 'GENERATION_REJECTED':
        from run_python_skill_cycle import contract_rejection_feedback
        # GENERATION_REJECTED also includes budget/network/provider errors.
        # Only a bound model response with a contract error is eligible.
        try:
            contract_rejection_feedback(folder, ledger)
        except (ValueError, OSError, KeyError, TypeError):
            return []
        return ['--generation-feedback-from', folder]
    return []


def execute_cycle(output, plan, policy_path, *, live=False, source_fixture=None, target_fixture=None,
                  lesson_learning=False, initial_lessons=None, python_revision_contract=False,
                  research_profile=None, release_certificate=None, skill_review_policy='legacy_v1'):
    from orchestrator.strategy_skill_review import LEGACY, validate_policy as validate_skill_review_policy
    skill_review_policy = validate_skill_review_policy(skill_review_policy)
    if skill_review_policy != LEGACY and not research_profile:
        raise ValueError('skill_review_requires_strategy_research_cycle')
    if python_revision_contract and (not lesson_learning or not live):
        raise ValueError('python_revision_contract_requires_live_lesson_cycle')
    release = None
    if research_profile:
        from orchestrator.research_profile import validate_profile
        validate_profile(research_profile)
        if not lesson_learning:
            raise ValueError('research_profile_requires_lesson_learning')
        if live:
            from orchestrator.research_release import validate_release_certificate
            if not release_certificate:
                raise ValueError('research_live_release_certificate_required')
            release = validate_release_certificate(release_certificate, root=ROOT)
    output = checked(output, True)
    plan = validate_plan(plan, research_profile=research_profile)
    if research_profile:
        # Admission observations exist before collection. Discover unsupported
        # clocks/provenance now, rather than after a paid source episode.
        from run_skill_discovery import source_case
        from tools.python_source_budget import observation_binding
        admission_bindings = {case['case_id']: observation_binding(source_case(case))
            for case in read(checked(plan['admission_manifest']))['cases'] if case['phase'] == 'admission'}
    policy_path = checked(policy_path)
    policy = validate_policy(read(policy_path))
    if live != policy["enabled"] or (not live and not (source_fixture and target_fixture)):
        raise ValueError("explicit_live_or_two_injected_fixtures_required")
    if live and (source_fixture or target_fixture):
        raise ValueError("live_cycle_cannot_inject_choices")
    if output.exists():
        raise ValueError("new_cycle_directory_required_no_partial_restart")
    output.mkdir(parents=True, exist_ok=False)
    hashes = runtime_hashes()
    write(output/"source_before.json", hashes)
    write(output/"precommitted_plan.json", plan)
    write(output/"policy.json", policy)
    if research_profile:
        write(output/'predeclared_admission_bindings.json', admission_bindings)
    role_paths = {}
    for role, config in policy["roles"].items():
        role_paths[role] = output/(role+"_config.json")
        write(role_paths[role], config)
    budget_path = output/"hierarchy.sqlite"
    budget = HierarchyBudget(budget_path, policy, transport_failure_policy=(
        'latch_on_error_v1' if research_profile else 'legacy_v1'))
    def command(label, tool, args):
        if research_profile:
            assert_cycle_sources(hashes)
        if research_profile and live:
            # A changed source tree invalidates admission before another request.
            validate_release_certificate(release_certificate, root=ROOT)
            if budget.transport_blocked():
                raise ValueError('research_cycle_transport_latched')
        command = [sys.executable, "-B", "-u", str(ROOT/"tools"/tool)] + [str(a) for a in args]
        result = run_recorded(command, ROOT, output/(label+"_command"))
        return result["exit_code"]
    def episode(name, task, memory, library, fixture, lessons=None):
        folder = output/name
        args = ["--run-dir", folder, "--mode", "llm" if live else "deterministic",
            "--hierarchy-policy", output/"policy.json", "--hierarchy-ledger", budget_path,
            "--planner-ledger", output/"planner.sqlite", "--ledger", output/"scout.sqlite",
            "--skill-library", library, "--provider-contract", "explicit_action_provider_v5",
            "--fallback-policy", "path_stage_v1", "--path-selection-policy", "path_probe_round_v1"]
        if research_profile:
            from orchestrator.research_profile import episode_flags
            # The profile supplies these options once, identically for both tasks.
            for flag in ('--provider-contract', '--fallback-policy', '--path-selection-policy'):
                index = args.index(flag)
                del args[index:index+2]
            args += episode_flags()
            if skill_review_policy != 'legacy_v1':
                args += ['--skill-review-policy', skill_review_policy]
            if live:
                args += ['--release-certificate', release_certificate]
        else:
            args += ["--memory-policy", "transfer_v1", "--memory-max-records", "4", "--stability-diagnostics"]
        if memory:
            args += ["--memory", memory]
        if lesson_learning:
            if not research_profile:
                args += ["--lesson-runtime", "--max-skill-attempts", "3"]
            if lessons:
                args += ["--lesson-library", lessons]
        if live:
            args += ["--allow-live-api", "--llm-config", role_paths["scout"]]
        else:
            args += ["--provider-fixture", checked(fixture)]
            if read(checked(library)).get("fixture"):
                args += ["--allow-fixture-library"]
        code = command(name, "run_hierarchical_episode.py", args + task_argv(task))
        if code != 0 or not (folder/"hierarchy_audit.json").is_file():
            raise RuntimeError("hierarchical_episode_process_failed")
        audit = read(folder/"audit.json")
        preserve = read(folder/"host_preservation.json")
        hierarchy = read(folder/"hierarchy_audit.json")
        if (not audit["validation_ok"] or not audit["actual_physics"]
                or not hierarchy["library_unchanged"] or not hierarchy.get("lesson_library_unchanged", True)
                or not all(preserve.values())):
            raise ValueError("hierarchical_episode_evidence_invalid")
        return dict(status=audit["status"], plan_success=audit["plan_success"],
            episode_dir=str(folder), hierarchy=hierarchy, artifacts={
                "report":str(folder/"worker_result.json"), "audit":str(folder/"hierarchy_audit.json"),
                "memory":str(folder/"action_memory_frozen.json")})

    initial_library = plan["initial_library"]
    if initial_library is None:
        initial_library = output/"empty_library.json"
        freeze_skill_union([], initial_library)
    inherited_lessons = []
    if initial_lessons:
        if not lesson_learning:
            raise ValueError("initial_lessons_require_opt_in")
        from orchestrator.lesson_memory import validate_library
        prior = validate_library(read(checked(initial_lessons)), read(checked(initial_library)), allow_fixture=not live)
        inherited_lessons = [e["lesson"] for e in prior["entries"]]

    def source(_):
        return episode("source_episode", plan["source"], plan["initial_memory"], initial_library, source_fixture,
                       initial_lessons)

    def prepare(state):
        source = state["source"]
        prep = prepare_episode_learning(source["episode_dir"], checked(plan["admission_manifest"]), output/"learning_inputs")
        memory = output/"memory_for_next_episode.json"
        frozen = freeze_cumulative_memory(([plan["initial_memory"]] if plan["initial_memory"] else [])
            + [source["artifacts"]["memory"]], [], memory)
        manifest = prep.get("manifest_path")
        # Keep the original preparation artifact. Make an explicit derived
        # Python synthesis manifest; admission data stay out of the model input.
        if manifest:
            prepared = read(Path(manifest))
            trigger = prep["selected"]["trigger_kind"]
            template = read(checked(plan["admission_manifest"]))
            if trigger == "stability_failed":
                prepared["python_failure_scope"] = "remaining_body_stability_failed"
                if not lesson_learning:
                    prepared["python_stability_pose_contract"] = True
                    prepared["python_support_preserve_contract"] = True
            elif trigger != "path_failed":
                raise ValueError("unsupported_python_learning_trigger")
            if lesson_learning:
                if template.get('python_applicability_cases'):
                    prepared['python_applicability_cases'] = deepcopy(template['python_applicability_cases'])
                admissions = [c for c in prepared['cases'] if c['phase'] == 'admission']
                if any(c['trigger_kind'] != trigger for c in admissions):
                    # Do not spend on a policy which cannot meet an unrelated
                    # predeclared gate. Preserve the selected observation.
                    return dict(status='UNSUPPORTED_VALIDATION_SCOPE', selected=prep['selected'],
                        memory=str(memory), record_count=frozen['record_count'], manifest=None,
                        reason='predeclared_admission_trigger_mismatch',
                        artifacts=dict(memory=str(memory), preparation=str(output/'learning_inputs/preparation.json')))
            prepared["derived_from_preparation_manifest_sha256"] = sha(Path(manifest))
            # Optional existing debug/code constraints must be declared before
            # this cycle, not chosen after seeing admission outcomes.
            prepared["hierarchy_declared_admission_template_sha256"] = sha(checked(plan["admission_manifest"]))
            manifest = output/"python_learning_manifest.json"
            write(manifest, prepared)
            if research_profile:
                packet, bindings = research_learning_packet(prepared, 'research-preflight:' + digest(str(output))[:16])
                write(output/'learning_observation_bindings.json', bindings)
                write(output/'learning_input_preflight.json', packet)
        return dict(status=prep["status"], selected=prep["selected"], memory=str(memory),
            record_count=frozen["record_count"], manifest=str(manifest) if manifest else None,
            artifacts=dict(memory=str(memory), preparation=str(output/"learning_inputs/preparation.json"),
                **({"manifest":str(manifest)} if manifest else {})))

    def learn(state):
        candidates, admitted, followup_args, failure = [], [], [], None
        lesson_path, lesson_result, consolidation_result = None, None, None
        ready = state["prepare"]["status"] == "PREPARED"
        if ready and lesson_learning:
            from run_python_skill_cycle import python_generation_packet
            from orchestrator.lesson_memory import with_prior_knowledge
            packet_path = output / "lesson_source_packet.json"
            manifest = read(Path(state["prepare"]["manifest"]))
            packet_id = "lesson-cycle:" + digest(str(output))[:16]
            packet = (research_learning_packet(manifest, packet_id)[0] if research_profile else
                      python_generation_packet(manifest, packet_id))
            packet = with_prior_knowledge(packet, initial_library, initial_lessons, allow_fixture=not live)
            write(packet_path, packet)
            args = ["--packet", packet_path, "--run-dir", output / "lesson_candidate",
                    "--mode", "live" if live else "fixture"]
            if research_profile:
                args += ['--transport-failure-policy', 'latch_on_error_v1']
            if live:
                args += ["--allow-live-api", "--config", role_paths["learner"],
                         "--ledger", output/"learner.sqlite", "--hierarchy-policy", output/"policy.json",
                         "--hierarchy-ledger", budget_path]
            code = command("lesson_extract", "run_lesson_discovery.py", args)
            if (output / "lesson_candidate/result.json").is_file():
                lesson_result = read(output / "lesson_candidate/result.json")
            ready = code == 0 and bool(lesson_result and lesson_result.get("status") == "CANDIDATE")
            if ready:
                lesson_path = lesson_result["lesson"]
            else:
                failure = "lesson_extraction_rejected_or_failed"
        if ready:
            for i in range(plan["max_proposals"] if live else 1):
                folder = output/("candidate_%02d"%(i+1))
                args = ["--run-dir", folder, "--manifest", state["prepare"]["manifest"],
                    "--mode", "live" if live else "fixture", "--generation-output-tokens", "4096"]
                if research_profile:
                    from orchestrator.research_profile import learning_flags
                    args += learning_flags()
                if live:
                    args += ["--allow-live-api", "--llm-config", role_paths["learner"],
                        "--ledger", output/"learner.sqlite", "--hierarchy-policy", output/"policy.json",
                        "--hierarchy-ledger", budget_path]
                args += followup_args
                if lesson_path:
                    args += ["--lesson", lesson_path]
                    if python_revision_contract:
                        args += ['--python-revision-contract']
                    args += ['--prior-library', initial_library]
                    if initial_lessons:
                        args += ['--prior-lessons', initial_lessons]
                code = command(folder.name, "run_python_skill_cycle.py", args)
                result_file = folder/"result.json"
                if not result_file.is_file():
                    failure = "learning_process_failed_before_result"
                    break
                result = read(result_file)
                candidates.append(dict(folder=str(folder), exit_code=code, result=result))
                if result.get("source_unchanged") is False:
                    raise ValueError("learning_source_changed")
                if research_profile and candidate_infrastructure_error(result):
                    failure = candidate_infrastructure_error(result)
                    break
                if candidate_admitted(result, code):
                    admitted.append(folder/"skills_frozen.json")
                    failure = None
                    break
                if result.get('admission', {}).get('admitted') is True:
                    failure = 'failed_candidate_process_cannot_publish_admission'
                    break
                # An explicit new candidate may use debug feedback only.
                # A bounded new candidate can use deterministic development
                # diagnostics. This never retries HTTP or increases max_proposals.
                followup_args = candidate_followup_args(folder, result, live=live,
                                                        ledger=output/'learner.sqlite')
                candidates[-1]['next_candidate_eligible'] = bool(followup_args)
                if not followup_args:
                    failure = 'learning_stopped:' + str(result.get('status'))
                    break
                failure = 'candidate_limit_exhausted:' + str(result.get('status'))
        if lesson_learning and admitted:
            args = ['--consolidate-library', admitted[0], '--draft', lesson_path,
                    '--run-dir', output/'lesson_finalized', '--mode', 'live' if live else 'fixture']
            if research_profile:
                args += ['--transport-failure-policy', 'latch_on_error_v1']
            if live:
                args += ['--allow-live-api', '--config', role_paths['learner'],
                         '--ledger', output/'learner.sqlite', '--hierarchy-policy', output/'policy.json',
                         '--hierarchy-ledger', budget_path]
            code = command('lesson_consolidate', 'run_lesson_discovery.py', args)
            if (output/'lesson_finalized/result.json').is_file():
                consolidation_result = read(output/'lesson_finalized/result.json')
            if code == 0 and consolidation_result and consolidation_result.get('status') == 'CANDIDATE':
                lesson_path = consolidation_result['lesson']
            else:
                # Keep technical admission artifacts, but do not publish a code
                # version with an unfinalized or rejected explanation.
                admitted = []
                lesson_path = None
                failure = 'post_validation_lesson_not_published'
        library = output/"library_for_next_episode.json"
        prior_snap = read(checked(initial_library))
        bases = [initial_library] if prior_snap["entries"] or not admitted else []
        conflict = None
        try:
            frozen = freeze_skill_union(bases + admitted, library, allow_fixture=not live)
        except ValueError as error:
            code = str(error)
            if not ("conflicting_frozen_skill_program" in code or "fixture" in code):
                raise
            conflict = "new_library_provenance_conflicts_with_prior"
            frozen = freeze_skill_union([initial_library], library, allow_fixture=not live)
        new_ids = [c["result"]["candidate_id"] for c in candidates
            if c["result"].get("admission",{}).get("admitted") and not conflict and admitted]
        lessons_path = None
        if lesson_learning:
            from orchestrator.lesson_memory import freeze_lessons
            lesson_candidates = inherited_lessons + ([read(Path(lesson_path))] if lesson_path else [])
            lesson_snapshot = freeze_lessons(lesson_candidates, frozen, allow_fixture=not live)
            lessons_path = output / "lessons_for_next_episode.json"
            write(lessons_path, lesson_snapshot)
        result = dict(status="REGISTERED" if new_ids else "NO_NEW_SKILL", candidates=candidates,
            new_ids=new_ids, library=str(library), library_entries=len(frozen["entries"]),
            learning_error=failure, merge_conflict=conflict, lesson_result=lesson_result,
            consolidation_result=consolidation_result,
            lessons=str(lessons_path) if lessons_path else None)
        write(output/"learning_result.json", result)
        return dict(result, artifacts={"library":str(library), "result":str(output/"learning_result.json")})

    def target(state):
        skip = target_skip_reason(state['learn'], research_profile)
        if skip:
            result = dict(status='NOT_RUN', plan_success=None, reason=skip,
                          automatic_calls=0, artifacts={})
            write(output/'target_not_run.json', result)
            return result
        return episode("next_episode", plan["target"], state["prepare"]["memory"],
            state["learn"]["library"], target_fixture, state["learn"].get("lessons"))

    def finish(state):
        target_ran = state['target']['status'] != 'NOT_RUN'
        memory_paths = [state['prepare']['memory']]
        if target_ran:
            memory_paths.append(state['target']['artifacts']['memory'])
        frozen = freeze_cumulative_memory(memory_paths,
            [], output/"memory_after_cycle.json")
        if lesson_learning:
            from orchestrator.lesson_memory import freeze_bundle
            freeze_bundle(output/"memory_after_cycle.json", state["learn"]["library"],
                          state["learn"]["lessons"], output/"knowledge_bundle.json", allow_fixture=not live)
        target_hierarchy = state['target'].get('hierarchy', {})
        used = target_hierarchy.get('physics_bindings', [])
        new_ids = set(state["learn"]["new_ids"])
        source_id = read(Path(state["source"]["artifacts"]["report"]))["run_id"]
        target_decisions = (read(Path(state["target"]["episode_dir"])/"hierarchy_decisions.json")
                            if target_ran else [])
        source_visible = source_cited = 0
        for d in target_decisions:
            ids = {m["evidence_id"] for m in d["request"]["memory_references"]
                   if m.get("source_run_id")==source_id}
            source_visible += bool(ids)
            source_cited += bool(ids & set((d["response"].get("choice") or {}).get("evidence_ids",[])))
        result = dict(status="COMPLETED" if target_ran else 'COMPLETED_WITH_TARGET_NOT_RUN', live_models=live,
            research_profile=research_profile, release_certificate=release,
            skill_review_policy=skill_review_policy,
            target_status=state['target']['status'], target_not_run_reason=state['target'].get('reason'),
            lesson_learning=lesson_learning, lesson_result=state["learn"].get("lesson_result"),
            post_validation_lesson=state['learn'].get('consolidation_result'),
            learning_error=state['learn'].get('learning_error'),
            target_lesson_supplies=target_hierarchy.get("lessons_supplied", 0),
            source_plan_success=state["source"]["plan_success"],
            target_plan_success=state["target"]["plan_success"],
            new_skill_ids=sorted(new_ids),
            next_episode_new_skill_attempts=sum(a["skill_id"] in new_ids for a in used),
            next_episode_new_skill_passes=sum(a["skill_id"] in new_ids and a["accepted"] for a in used),
            source_memory_visible_in_target=source_visible,
            source_memory_cited_in_target=source_cited,
            any_frozen_memory_visible_in_target=target_hierarchy.get("memory_visible_requests", 0),
            final_memory_records=frozen["record_count"], api_budget=budget.summary(),
            source_unchanged=all(sha(ROOT/n)==h for n,h in hashes.items()),
            performance_benefit="NOT_RUN", interrupted_solver_resume=False,
            complete_lifecycle_does_not_imply_skill_or_plan_success=True)
        write(output/"result.json", result)
        return dict(result, artifacts={"result":str(output/"result.json"),
            "memory":str(output/"memory_after_cycle.json")})
    identity = dict(plan=plan, policy=policy, source_hashes=hashes, lesson_learning=lesson_learning,
        initial_lessons_sha256=sha(checked(initial_lessons)) if initial_lessons else None,
        source_fixture_sha256=sha(checked(source_fixture)) if source_fixture else None,
        target_fixture_sha256=sha(checked(target_fixture)) if target_fixture else None)
    if research_profile:
        from orchestrator.research_profile import EPISODE_SETTINGS
        identity['research_profile'] = dict(profile_id=research_profile, policy=deepcopy(EPISODE_SETTINGS),
            observation_contract='source_bound_v1', release_certificate=release)
    if python_revision_contract:
        identity['python_revision_contract_version'] = 1
    if skill_review_policy != 'legacy_v1':
        identity['skill_review_policy'] = skill_review_policy
    graph = LearningCycleWorkflow(output, identity, dict(source=source, prepare=prepare, learn=learn, target=target, finish=finish))
    result = graph.run()
    write(output/"graph_result.json", result)
    return read(output/"result.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--hierarchy-policy", required=True, type=Path)
    p.add_argument("--allow-live-api", action="store_true")
    p.add_argument("--source-fixture", type=Path)
    p.add_argument("--target-fixture", type=Path)
    p.add_argument("--lesson-learning", action="store_true")
    p.add_argument('--python-revision-contract', action='store_true')
    p.add_argument("--initial-lessons", type=Path)
    from orchestrator.research_profile import PROFILE_ID
    p.add_argument('--research-profile', choices=(PROFILE_ID,))
    p.add_argument('--release-certificate', type=Path)
    p.add_argument('--skill-review-policy', choices=('legacy_v1', 'skill_selection_review_v1'),
                   default='legacy_v1')
    a = p.parse_args(argv)
    result = execute_cycle(a.run_dir, read(checked(a.plan)), a.hierarchy_policy,
        live=a.allow_live_api, source_fixture=a.source_fixture, target_fixture=a.target_fixture,
        lesson_learning=a.lesson_learning, initial_lessons=a.initial_lessons,
        python_revision_contract=a.python_revision_contract, research_profile=a.research_profile,
        release_certificate=a.release_certificate, skill_review_policy=a.skill_review_policy)
    print(json.dumps(result), flush=True)
    return 0 if result["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
