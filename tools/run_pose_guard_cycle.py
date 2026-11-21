"""Focused learned guard recovery, scoped admission, free selection and ablation.

Default mode only prepares/estimates. Live requires a new explicit authorization.
This is a declared eight-part state replay, NOT whole-product DFS evaluation.
"""
from copy import deepcopy
from decimal import Decimal
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "worker"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from orchestrator.conditional_plan import action_for_step
from orchestrator.explicit_action_memory import append_action_observation, freeze_memory
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchy_budget import HierarchyBudget, estimate_request, validate_policy
from orchestrator.llm_ranker import DEFAULTS, _json
from orchestrator.pose_guard_case import prepare
from orchestrator.pose_guard_discovery import PoseGuardSkillDiscoveryService
from orchestrator.pose_guard_recovery import KIND, guard_packet, make_guard
from orchestrator.pose_guard_registry import PoseGuardSkillRegistry, contract_probe
from orchestrator.pose_guard_runtime import step_guard_program
from orchestrator.skill_registry import retrieve
from orchestrator.skill_validation import _actual_attempt
from worker.explicit_action_contract import digest, validate_request
from worker.task_memory import _safe_path, _lexical_path
from run_recorded import run_recorded

BODY_CAP = 200000
MAX_PHYSICS = 9


def read(path):
    return json.loads(_safe_path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    with _safe_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def linux(path):
    # Executable source lives outside runs; data still uses _safe_path.
    # Reject forbidden drives/traversal BEFORE absolute/lstat conversion.
    from worker.remote_rank_client import checked_path
    candidate = Path(_lexical_path(path)).absolute()
    if not candidate.is_relative_to(ROOT):
        raise ValueError("worker_source_must_stay_in_project")
    value = str(checked_path(candidate, run_only=False)).replace("\\", "/")
    return "/mnt/c/" + value[3:]


def policy(enabled=False):
    def cfg(model, calls, usd, inp, out, effort):
        return dict(DEFAULTS, enabled=True, model=model, max_calls=calls, max_usd=usd,
                    input_usd_per_million=inp, output_usd_per_million=out,
                    max_output_tokens=4096, reasoning_effort=effort)
    # Reuse the existing immutable multi-model quota implementation. Its scout
    # slot carries Luna generation here; there is no scouting/disassembly phase.
    return validate_policy(dict(schema_version=1, enabled=enabled, max_calls=7,
        max_usd="2.00", roles=dict(
            scout=cfg("gpt-6-luna", 3, ".10", ".10", ".50", "none"),
            planner=cfg("gpt-6.1-sol", 4, "1.90", "2", "10", "low"),
            learner=cfg("gpt-6-luna", 0, "0", ".10", ".50", "none")),
        episode=dict(scout_calls=0, planner_calls=4, fallback_calls=0,
                     max_plan_steps=1, recovery_per_step=1)))


class BoundedDiscovery(PoseGuardSkillDiscoveryService):
    def _body(self, envelope):
        body = super()._body(envelope)
        if len(_json(body).encode("utf-8")) > BODY_CAP:
            raise ValueError("focused_request_body_cap")
        return body


class BoundedSol(SolPlanService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, provider_contract="conditional_plan_v3",
                         planning_feedback_policy="path_evidence_v1", **kwargs)

    def _body(self, envelope):
        body = super()._body(envelope)
        if len(_json(body).encode("utf-8")) > BODY_CAP:
            raise ValueError("focused_request_body_cap")
        return body


def escalation(bundle, earlier=None):
    return dict(reason="recorded_plan_guard_recovery", guard=deepcopy(bundle["guard"]),
        scope="One fresh removal from the declared previously reached state. Not a full plan.",
        goal="Independently select a legal next action or registered program. Selection is not forced.",
        guard_semantics="The old planned action was never physically executed. A guard is not collision.",
        original_planned_mover=bundle["guard"]["intended"]["part_id"],
        alternative_mover_is_progress_not_original_mover_repair=True,
        physical_repositioning_trajectory_verified=False,
        shared_build_observations=deepcopy(bundle.get("shared_build_observations", [])),
        shared_build_observation_rule="Historical actual measurements for BOTH arms; not proof of current success. Cite action evidence only when present in observation.evidence_ids.",
        earlier_decisions=deepcopy(earlier or []))


def envelope(bundle, cards, earlier=None):
    return dict(request=bundle["request"], skills=cards, escalation=escalation(bundle, earlier),
        max_steps=1, timeout_seconds=90, plan_provider_contract="conditional_plan_v3",
        planning_feedback_policy="path_evidence_v1")


def summarize(raw):
    return {key: deepcopy(raw.get(key)) for key in ("status", "path_success",
        "stability_success", "chosen", "path_check_calls", "stability_check_calls",
        "elapsed_seconds", "backend", "endpoint_observation", "stability_diagnostics")}


def context_for(source, request, response):
    context = deepcopy(source["context"])
    context.update(pose_catalog=deepcopy(request["pose_catalog"]),
        remaining_parts=deepcopy(request["remaining_parts"]), event_id=request["request_id"],
        action_id=response["action_id"], remaining_wall_seconds=65, stability_timeout=65)
    choice = response["choice"]
    context["parts_rest"] = [p for p in request["remaining_parts"] if p != choice["part_id"]]
    context["pose_matrix"] = next(p["matrix"] for p in request["pose_catalog"]
                                 if p["pose_id"] == choice["pose_id"])
    context.pop("time_remaining_seconds", None)
    return context


class Physics:
    def __init__(self, source, root):
        self.source, self.root = source, root
        self.calls, self.started = 0, time.monotonic()

    def check_time(self):
        # Leave enough time for a bounded 90 s provider or 80 s worker call.
        if time.monotonic() - self.started >= 1080:
            raise RuntimeError("focused_twenty_minute_limit")

    def run(self, bundle, response, folder):
        self.check_time()
        if self.calls >= MAX_PHYSICS:
            raise RuntimeError("focused_physical_budget")
        folder = _safe_path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        req = bundle["request"]
        packet = dict(context=context_for(self.source, req, response), choice=response["choice"])
        save(folder / "input.json", packet)
        argv = ["C:/Windows/System32/wsl.exe", "-d", "ASAP-Ubuntu", "-u", "root",
            "--cd", "/opt/asap-research/ASAP", "--", "/usr/bin/timeout",
            "--signal=TERM", "--kill-after=5s", "80s",
            "/opt/asap-research/envs/asap/bin/python", "-B", "-u",
            linux(ROOT / "worker/run_pose_guard_action.py"),
            "--input", linux(folder / "input.json"), "--output", linux(folder / "physics")]
        self.calls += 1
        execution = run_recorded(argv, ROOT, folder / "execute")
        if execution["exit_code"] != 0:
            raise RuntimeError("fresh_physics_process_failed")
        expected = {k: self.source["reference_raw"][k] for k in ("input_hashes", "source_hashes")}
        evidence = _actual_attempt(folder / "physics", expected, False, False)
        raw = evidence["result"]["value"]
        observation = dict(status=raw["status"], path_success=raw["path_success"],
            stability_success=raw["stability_success"], verifier_origin=raw["backend"],
            raw=raw, evidence_ref=str(folder / "physics/result.json"))
        append_action_observation(self.root / "new_experience.jsonl", req, response, observation)
        save(folder / "verification.json", dict(actual_asap_artifacts_checked=True,
            late=evidence["late"], accepted_local_removal=raw["status"] == "PASS" and not evidence["late"],
            whole_assembly_success=False))
        if evidence["late"]:
            return dict(raw, status="LATE")
        return raw


def bundle_file(source, source_path, run_id, yaw, path):
    bundle = prepare(source, run_id, yaw)
    bundle.update(source_failure_path=str(source_path), source_failure_sha256=sha(source_path))
    save(path, bundle)
    return bundle


def skill_cards(rows, bundle):
    cards = []
    for row in rows:
        run = step_guard_program(row["program"], bundle["request"], bundle["guard"],
                                 bundle["plan"], bundle["index"])
        cards.append(dict(skill_id=row["strategy_id"], language="python_policy_v1",
            description=row["card"]["description"], program=row["program"],
            lifecycle=row["lifecycle"], validation_family=KIND,
            validation_scope="Same recorded eight-part state; declared world-yaw cases. Not unseen geometry.",
            current_proposal=(run["response"] or {}).get("choice"),
            applicability=run["not_applicable_reason"],
            current_attempt_verified=False, requires_fresh_physics=True))
    return cards


def build_observation_view(snapshot):
    """Same compact preparation facts for both arms; no generated code/prose."""
    return [dict(evidence_id=r["evidence_id"], source_run_id=r["source_run_id"],
        remaining_parts=r["scope"]["remaining_parts"],
        choice=deepcopy(r["choice"]), pose_matrix=deepcopy(r["scope"]["pose_matrix"]),
        status=r["result"]["status"], path_success=r["result"]["path_success"],
        stability_success=r["result"]["stability_success"],
        deadline_exceeded=(r["result"].get("raw", {}).get("elapsed_seconds", 0) > 65),
        evidence_ref=r["result"]["evidence_ref"], current_result_not_implied=True)
        for r in snapshot["records"]]


def feedback_after(bundle, response, raw, event_id):
    value = deepcopy(bundle)
    request = value["request"]
    request["request_id"] = event_id + ":decision"
    identity = response["action_id"]
    choice = deepcopy(response["choice"])
    request["attempted_action_ids"].append(identity)
    request.setdefault("attempted_actions", []).append(dict(action_id=identity,
        **{k: v for k, v in choice.items() if k != "evidence_ids"}))
    request.setdefault("path_stage_observations", []).append(dict(action_id=identity,
        path_success=raw["path_success"], status=raw["status"],
        deadline_exceeded=raw["elapsed_seconds"] > 65, backend=raw["backend"],
        source_event_id=event_id))
    request["episode_feedback"].append(dict(evidence_id=event_id, action_id=identity,
        status=raw["status"], choice=choice, remaining_parts=request["remaining_parts"],
        selected_pose_matrix=next(p["matrix"] for p in request["pose_catalog"]
                                 if p["pose_id"] == choice["pose_id"]),
        path_success=raw["path_success"], stability_success=raw["stability_success"],
        accepted_for_transition=False, deadline_exceeded=raw["elapsed_seconds"] > 65))
    request["episode_feedback"] = request["episode_feedback"][-12:]
    request["evidence_ids"] = sorted({r["evidence_id"] for r in
        request["episode_feedback"] + request["memory_references"]})
    validate_request(request)
    value["guard"] = make_guard(value["plan"], value["index"], request)
    return value


def evaluate_arm(source, initial, rows, service, physics, folder):
    bundle, history = deepcopy(initial), []
    # Different request identity avoids cross-arm ledger/cache collisions while
    # leaving physical state, frozen observations, seed and limits identical.
    bundle["request"]["request_id"] += ":" + folder.name
    bundle["guard"] = make_guard(bundle["plan"], bundle["index"], bundle["request"])
    for index in range(2):
        physics.check_time()
        cards = skill_cards(rows, bundle)
        save(folder / ("decision_%d_input.json" % index), envelope(bundle, cards, history))
        result = service.plan(bundle["request"], cards, escalation(bundle, history), max_steps=1)
        save(folder / ("decision_%d_provider.json" % index), result)
        if result["status"] != "planned" or not result["plan"]["steps"]:
            return dict(status="MODEL_PLAN_REJECTED_OR_EMPTY", history=history, reused=False,
                fallback_reason=result.get("fallback_reason"),
                provider_status_code=result.get("metadata", {}).get("status_code"),
                api_attempted=result.get("metadata", {}).get("attempted", False))
        plan, selected_id = result["plan"], None
        step = plan["steps"][0]
        try:
            if step["mode"] == "registered_skill":
                selected_id = step["registered_skill_id"]
                row = next(r for r in rows if r["strategy_id"] == selected_id)
                response = step_guard_program(row["program"], bundle["request"], bundle["guard"],
                                              bundle["plan"], bundle["index"])["response"]
                if response is None or response["choice"]["part_id"] != step["part_id"]:
                    raise ValueError("selected_program_abstained_or_mover_mismatch")
            else:
                response = action_for_step(plan, 0, bundle["request"])
        except (ValueError, StopIteration) as error:
            history.append(dict(selection=selected_id, execution="guard_refused", reason=str(error)))
            bundle["request"]["request_id"] += ":after_refusal"
            bundle["guard"] = make_guard(bundle["plan"], bundle["index"], bundle["request"])
            continue
        save(folder / ("decision_%d_action.json" % index), response)
        raw = physics.run(bundle, response, folder / ("attempt_%d" % index))
        item = dict(selection=selected_id, outcome=summarize(raw),
                    program_executed=selected_id is not None)
        history.append(item)
        if raw["status"] == "PASS":
            return dict(status="PASS_ONE_REMOVAL", reused=selected_id is not None, history=history,
                selected_skill_id=selected_id,
                original_planned_mover_removed=response["choice"]["part_id"] == initial["guard"]["intended"]["part_id"],
                whole_assembly_success=False)
        if raw["status"] in ("ERROR", "LATE"):
            break
        bundle = feedback_after(bundle, response, raw, bundle["request"]["run_id"] + ":failed:%d" % index)
    return dict(status="NOT_RESOLVED_WITHIN_TWO_DECISIONS", reused=False, history=history)


def cycle_status(with_skill, without_skill):
    if not with_skill.get("reused"):
        return "REGISTERED_BUT_REUSE_NOT_PROVEN"
    if not isinstance(without_skill, dict):
        return "REUSED_CONTROL_NOT_RUN"
    if without_skill.get("fallback_reason") in ("api_error", "api_timeout", "unknown_outcome"):
        return "REUSED_CONTROL_API_BLOCKED"
    if not any("outcome" in row for row in without_skill.get("history", [])):
        return "REUSED_CONTROL_NO_PHYSICS"
    return "REUSED_AND_CONTROL_EXECUTED"


def preview(source, source_path, output):
    cfg = policy()
    bundle = bundle_file(source, source_path, "pose-guard-preflight", 0, output / "debug_bundle.json")
    service = BoundedDiscovery(cfg["roles"]["scout"], output / "preview_luna.sqlite")
    packet = guard_packet(bundle["guard"], bundle["request"], "pose-guard-preflight")
    estimate = service.preview(packet)
    save(output / "generation_preview.json", estimate)
    planner = BoundedSol(cfg["roles"]["planner"], output / "preview_sol.sqlite")
    body = planner._body(envelope(bundle, []))
    save(output / "planning_preview_without_skill.json", body)
    maxima = {}
    for role in ("scout", "planner"):
        c = cfg["roles"][role]
        maxima[role] = str(((BODY_CAP + c["request_overhead_tokens"]) *
            Decimal(c["input_usd_per_million"]) + c["max_output_tokens"] *
            Decimal(c["output_usd_per_million"])) / 1000000)
    upper = 3 * Decimal(maxima["scout"]) + 4 * Decimal(maxima["planner"])
    assert upper <= Decimal(cfg["max_usd"])
    result = dict(status="READY_FOR_BUDGET_APPROVAL", api_calls=0, key_accessed=False,
        max_calls=7, max_usd=cfg["max_usd"], model_roles={"Luna_generation": 3, "Sol_with_skill": 2, "Sol_without_skill": 2},
        generation_body_bytes=estimate["body_bytes"], generation_reservation=estimate["cost_estimate"],
        planner_without_skill_estimate=estimate_request(cfg, "planner", body),
        per_request_body_cap=BODY_CAP, max_reservations_per_call=maxima,
        all_seven_maximum_reservation=str(upper), auto_retries=0,
        physical_attempt_cap=MAX_PHYSICS, physical_seconds_per_attempt=65, overall_minutes=20,
        development_yaw=[0], admission_yaw=[90, 180], evaluation_yaw=0,
        original_part_feasibility_separate=True, original_failure_before_physics=True,
        comparison="New generated skill enabled versus no generated skills; same frozen memory/guard.",
        all_live_stages="NOT_RUN")
    save(output / "policy_disabled.json", cfg)
    save(output / "preflight.json", result)
    print(_json(result), flush=True)


def services(pool, cfg, ledger_root):
    generation_transport = pool.transport("scout")
    planning_transport = pool.transport("planner")
    generator = BoundedDiscovery(cfg["roles"]["scout"], ledger_root / "luna.sqlite", generation_transport)
    planner = BoundedSol(cfg["roles"]["planner"], ledger_root / "sol.sqlite", planning_transport)
    # A quota wrapper is an injected callable but may perform actual HTTP.
    # Preserve its verified mode, as the existing hierarchical runner does.
    generator._live_transport = generation_transport.live
    planner._live_transport = planning_transport.live
    return generator, planner


def recover_completed_generation(prior, generator, pool):
    from orchestrator.pose_guard_provenance import verified_proposal
    result = read(prior / "result.json")
    if result["status"] != "GENERATION_REJECTED" or result["physical_attempts"] != 0:
        raise ValueError("only_unused_bridge_rejection_can_resume")
    packet = read(prior / "revision_0/generation_packet.json")
    proposal = read(prior / "revision_0/provider_proposal.json")
    env = generator._envelope(packet)
    body = generator._body(env)
    with sqlite3.connect("file:" + (prior / "luna.sqlite").as_posix() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        local = db.execute("SELECT * FROM rank_requests WHERE request_id=?", (packet["request_id"],)).fetchone()
    matches = [r for r in pool.summary()["calls"] if r["id"] == digest(dict(role="scout", body=body))]
    if local is None or len(matches) != 1:
        raise ValueError("bound_live_bridge_rows_required")
    return packet, verified_proposal(proposal, dict(local), matches[0], env, body)


def live(source, source_path, output, resume_from=None):
    cfg = policy(True)
    ledger_root = _safe_path(resume_from) if resume_from is not None else output
    if resume_from is not None and read(ledger_root / "policy.json") != cfg:
        raise ValueError("resume_must_preserve_original_budget_policy")
    save(output / "authorization.json", dict(max_calls=7, max_usd=cfg["max_usd"],
        authorization_source="explicit_user_approval_then_allow_live_flag", auto_retries=0,
        continuation_of=str(resume_from) if resume_from else None,
        shared_original_ledger=str(ledger_root / "global_budget.sqlite")))
    save(output / "policy.json", cfg)
    pool = HierarchyBudget(ledger_root / "global_budget.sqlite", cfg)
    generator, planner = services(pool, cfg, ledger_root)
    recovered = recover_completed_generation(ledger_root, generator, pool) if resume_from else None
    physics = Physics(source, output)
    status = dict(status="STARTED", registered=False, reused=False, without_skill="NOT_RUN",
                  whole_assembly_success=False, unseen_generalization=False)
    packet, previous = None, None
    try:
        from local_api_key import key_environment
        with key_environment():
            for revision in range(3):
                physics.check_time()
                folder = output / ("revision_%d" % revision)
                replay_id = (read(ledger_root / "revision_0/debug_bundle.json")["request"]["run_id"]
                             if recovered and revision == 0 else output.name + ":debug:%d" % revision)
                bundle = bundle_file(source, source_path, replay_id,
                                     0, folder / "debug_bundle.json")
                packet = guard_packet(bundle["guard"], bundle["request"], output.name + ":%d" % revision)
                packet["timeout_seconds"] = 90
                if previous:
                    packet.update(operation="refine", previous_program=previous["program"],
                                  validation_feedback=[previous["feedback"]])
                    packet.pop("existing_skills")
                if recovered and revision == 0:
                    packet = recovered[0]
                save(folder / "generation_packet.json", packet)
                proposal = recovered[1] if recovered and revision == 0 else generator.propose(packet)
                save(folder / "provider_proposal.json", proposal)
                if proposal["status"] != "proposed" or proposal["origin"] != "actual_llm":
                    status["status"] = "GENERATION_REJECTED"
                    break
                program = proposal["program"]
                if previous and program == previous["program"]:
                    status["status"] = "UNCHANGED_FAILED_PROGRAM"
                    break
                card = deepcopy(proposal["card"])
                card["description"] = ("Unverified provider description: " + card["hypothesis"]
                    + "\nActual guarded Python program:\n" + program["source"])
                registry = PoseGuardSkillRegistry(folder / "registry")
                candidate = registry.propose(program, card, dict(origin="live_llm",
                    model=cfg["roles"]["scout"]["model"], generation_id=packet["request_id"],
                    provider_response_path=str(folder / "provider_proposal.json"),
                    provider_response_sha256=sha(folder / "provider_proposal.json")))
                identity = candidate["candidate_id"]
                cases = []
                for name, phase, yaw in (("debug", "debug", 0), ("yaw90", "admission", 90),
                                         ("yaw180", "admission", 180)):
                    if name != "debug":
                        bundle_file(source, source_path, output.name + ":" + name, yaw,
                                    folder / (name + "_bundle.json"))
                    cases.append(dict(case_id=name, phase=phase, source_task="01256",
                        program_hash=identity, trigger_kind=KIND,
                        guard_bundle_sha256=sha(folder / (name + "_bundle.json")),
                        expected_attempt_dir=str(folder / name / "physics"),
                        **{k: source["reference_raw"][k] for k in ("input_hashes", "source_hashes")}))
                registry.set_manifests(identity, dict(phase="debug", cases=cases[:1]),
                                       dict(phase="admission", cases=cases[1:]))
                records = []
                for case in cases:
                    name = case["case_id"]
                    current = read(folder / (name + "_bundle.json"))
                    selected = step_guard_program(program, current["request"], current["guard"],
                                                  current["plan"], current["index"])
                    probe = contract_probe(program, current)
                    save(folder / (name + "_contract.json"), probe)
                    save(folder / (name + "_selection.json"), dict(program_hash=identity,
                        guard_bundle_sha256=case["guard_bundle_sha256"], response=selected["response"]))
                    attempt = None
                    if selected["response"] is not None:
                        print(_json(dict(stage="actual_physics", revision=revision, case=name)), flush=True)
                        physics.run(current, selected["response"], folder / name)
                        attempt = case["expected_attempt_dir"]
                    artifacts = dict(guard_bundle_path=str(folder / (name + "_bundle.json")),
                        selection_path=str(folder / (name + "_selection.json")), attempt_dir=attempt)
                    record = registry.add_validation(identity, case, artifacts)
                    records.append(record)
                    ok_contract = all(probe[k] for k in ("static_contract", "not_applicable",
                        "duplicate_action", "invalid_output_rejected", "observation_sensitivity", "no_literal_source_ids"))
                    if record["outcome"] != "PASS" or not ok_contract:
                        if name == "debug":
                            previous = dict(program=program, feedback=dict(phase="debug",
                                case_id="debug", outcome=record["outcome"], contract=probe,
                                selected_choice=(selected["response"] or {}).get("choice"),
                                actual_physics=summarize(read(Path(attempt) / "result.json")) if attempt else None))
                        break
                decision = registry.admit(identity)
                save(folder / "decision.json", decision)
                if decision["admitted"]:
                    snapshot = registry.freeze(folder / "skills_frozen.json")
                    status["registered"] = True
                    # The identical initial bundle is passed by value to BOTH
                    # arms. No on-arm outcome or newly collected memory flows off.
                    initial = bundle_file(source, source_path, output.name + ":evaluation", 0,
                                          output / "evaluation_frozen_input.json")
                    build_memory = freeze_memory(output / "new_experience.jsonl",
                                                  output / "build_memory_frozen.json")
                    initial["shared_build_observations"] = build_observation_view(build_memory)
                    save(output / "evaluation_frozen_input_with_shared_experience.json", initial)
                    rows = retrieve(snapshot)
                    status["with_skill"] = evaluate_arm(source, initial, rows, planner, physics, output / "with_skill")
                    status["reused"] = status["with_skill"].get("reused", False)
                    if status["reused"]:
                        status["without_skill"] = evaluate_arm(source, initial, [], planner, physics, output / "without_skill")
                    status["status"] = cycle_status(status["with_skill"], status["without_skill"])
                    break
                status["status"] = "NOT_REGISTERED"
                if len(records) > 1:
                    # Admission feedback is held out. No refinement from it.
                    break
    except Exception as error:
        # Exception text may originate at the provider. Save only its type here;
        # service/worker logs already retain their bounded structured diagnostics.
        status.update(status="ERROR", error_type=type(error).__name__)
        raise
    finally:
        status.update(physical_attempts=physics.calls, budget=pool.summary())
        if (output / "new_experience.jsonl").exists():
            freeze_memory(output / "new_experience.jsonl", output / "new_experience_frozen.json")
        save(output / "result.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preview", "live"), default="preview")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--resume-completed-generation", type=Path)
    args = parser.parse_args()
    if args.mode == "live" and not args.allow_live:
        raise ValueError("explicit_new_seven_call_2_00_approval_required")
    source_path, output = _safe_path(args.source), _safe_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = read(source_path)
    if args.mode == "live":
        live(source, source_path, output, args.resume_completed_generation)
    else:
        if args.resume_completed_generation is not None:
            raise ValueError("resume_is_live_only")
        preview(source, source_path, output)


if __name__ == "__main__":
    main()
