"""Prepare or generate a lesson; never execute physics or automatically spend."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from run_integrated_repair import checked
from run_luna_experiment import read, write, sha
from run_skill_discovery import preview_config
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.lesson_memory import evidence_view, make_lesson


def extract(packet, output, config, *, mode="preview", ledger=None, transport=None):
    output = checked(output, True)
    output.mkdir(parents=True, exist_ok=False)
    packet = deepcopy(packet)
    packet["request_id"] += ":lesson"
    stage = packet['context'].get('lesson_stage', 'extract')
    if stage not in ('extract', 'consolidate'):
        raise ValueError('lesson_extraction_or_consolidation_required')
    packet["context"]["lesson_stage"] = stage
    astra = config.get('model') == 'gpt-6-astra'
    packet["context"]["python_output_tokens"] = config["max_output_tokens"] if astra else min(2048, config["max_output_tokens"])
    if astra:
        packet['timeout_seconds'] = 600
    write(output / "packet.json", packet)
    service = LessonDiscoveryService(config, ledger or output / "ledger.sqlite", transport)
    if transport is not None:
        service._live_transport = transport.live
    try:
        preview = service.preview(packet)
        write(output / "preview.json", preview)
        if mode == "preview":
            return dict(status="PREVIEW", api_calls=0)
        if mode == "fixture":
            facts = evidence_view(packet)
            fact = next((f for f in facts if f["observed_kind"] != "unknown"), facts[0])
            claim = dict(title="HANDWRITTEN wiring lesson", hypothesis="Fixture only; no inferred physical cause.",
                when_to_apply="Only the supplied observed failure stage.",
                when_not_to_apply="No bound current failure or different conditions.",
                evidence_ids=[fact["evidence_id"]], counterexample_ids=[],
                roles=["current mover and current legal supports"], trigger_kind=fact["observed_kind"])
            if stage == 'consolidate':
                claim.update(packet['context']['repair_binding']['contract'])
                claim['evidence_ids'] = [f['evidence_id'] for f in facts]
                claim['title'] = 'HANDWRITTEN post-validation summary fixture'
            response = dict(status="proposed", origin="manual_authored", lesson=claim)
        elif mode == "live":
            from run_python_skill_cycle import require_live_config
            from local_api_key import key_environment
            require_live_config(config, service)
            with key_environment():
                response = service.propose(packet)
        else:
            raise ValueError("invalid_lesson_mode")
        write(output / "response.json", response)
        if response.get("status") != "proposed":
            result = dict(status="LESSON_REJECTED", reason=response.get("fallback_reason"))
            from orchestrator.lesson_diagnostics import save_rejection_artifacts
            artifacts = save_rejection_artifacts(output, response)
            if artifacts:
                result['rejection_artifacts'] = artifacts
                result['diagnostic'] = response.get('metadata', {}).get('lesson_validation_diagnostics')
        else:
            provenance = dict(origin=response["origin"],
                response_path=str(output / "response.json"), response_sha256=sha(output / "response.json"),
                packet_path=str(output / "packet.json"), packet_sha256=sha(output / "packet.json"))
            if stage == 'consolidate':
                from orchestrator.lesson_consolidation import make_final_lesson
                lesson = make_final_lesson(response['lesson'], packet, provenance)
            else:
                lesson = make_lesson(response["lesson"], packet, provenance)
            write(output / "lesson.json", lesson)
            result = dict(status="CANDIDATE", lesson=str(output / "lesson.json"),
                          lesson_id=lesson["content_sha256"], actual_model=mode == "live")
        write(output / "result.json", result)
        return result
    finally:
        service.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path)
    p.add_argument('--consolidate-library', type=Path)
    p.add_argument('--draft', type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("preview", "fixture", "live"), default="preview")
    p.add_argument("--config", type=Path)
    p.add_argument("--ledger", type=Path)
    p.add_argument("--hierarchy-policy", type=Path)
    p.add_argument("--hierarchy-ledger", type=Path)
    p.add_argument('--transport-failure-policy', choices=('legacy_v1', 'latch_on_error_v1'), default='legacy_v1')
    p.add_argument("--allow-live-api", action="store_true")
    a = p.parse_args(argv)
    live = a.mode == "live"
    if live != a.allow_live_api or (live and not all((a.config, a.ledger, a.hierarchy_policy, a.hierarchy_ledger))):
        raise ValueError("explicit_approved_lesson_budget_required")
    config = read(checked(a.config)) if a.config else dict(preview_config(), max_output_tokens=4096)
    if not live and (config["enabled"] or config["max_calls"] or config["max_usd"]):
        raise ValueError("offline_requires_disabled_config")
    transport = None
    if live:
        from orchestrator.hierarchy_budget import HierarchyBudget
        policy = read(checked(a.hierarchy_policy))
        if not policy["enabled"] or config != policy["roles"]["learner"]:
            raise ValueError("lesson_learner_role_mismatch")
        transport = HierarchyBudget(checked(a.hierarchy_ledger, True), policy,
            transport_failure_policy=a.transport_failure_policy).transport("learner")
    if a.consolidate_library:
        if a.packet or not a.draft:
            raise ValueError('consolidation_requires_draft_and_no_raw_packet')
        from orchestrator.lesson_consolidation import repair_packet
        packet = repair_packet(checked(a.draft), checked(a.consolidate_library),
            'finalize:' + str(a.run_dir.name), allow_fixture=not live)
    else:
        if not a.packet or a.draft:
            raise ValueError('lesson_packet_required')
        packet = read(checked(a.packet))
    result = extract(packet, a.run_dir, config, mode=a.mode,
                     ledger=checked(a.ledger, True) if a.ledger else None, transport=transport)
    print(result)
    return 1 if result["status"] == "LESSON_REJECTED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
