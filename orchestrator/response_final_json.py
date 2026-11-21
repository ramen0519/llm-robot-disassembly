"""Strict Responses final JSON decoder with text-free diagnostic codes."""
import hashlib
import json
from orchestrator.llm_ranker import MAX_RESPONSE_BYTES


def decode_final_json(raw, diagnostics):
    """Accept one unambiguous final message, never merge commentary and JSON."""
    diagnostics.update(schema_version=1, contract="phase_final_json_v1",
        phase="decode", code="not_decoded", raw_response_recorded=False,
        provider_text_recorded=False, ignored_commentary_messages=0)

    def reject(code):
        diagnostics["code"] = code
        raise ValueError(code)

    if type(raw) is not dict:
        reject("invalid_response_object")
    status = raw.get("status")
    diagnostics["response_status"] = status if status in ("completed", "incomplete", "failed") else "unknown"
    if status != "completed":
        reject("llm_incomplete" if status == "incomplete" else "api_error")
    if raw.get("error") is not None or type(raw.get("output")) is not list:
        reject("invalid_output_container")
    diagnostics["output_item_count"] = len(raw["output"])
    messages, phases, total_bytes = [], [], 0
    for item in raw["output"]:
        if type(item) is not dict:
            reject("invalid_output_item")
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            reject("unexpected_output_item")
        if item.get("status") not in (None, "completed"):
            reject("incomplete_message")
        phase = item.get("phase")
        phases.append(phase if phase in (None, "commentary", "final_answer") else "unknown")
        diagnostics["message_phases"] = list(phases)
        if phase not in (None, "commentary", "final_answer"):
            reject("unknown_message_phase")
        if type(item.get("content")) is not list:
            reject("invalid_message_content")
        texts = []
        for block in item["content"]:
            if type(block) is not dict:
                reject("invalid_content_block")
            if block.get("type") == "refusal":
                reject("llm_refusal")
            if block.get("type") != "output_text" or type(block.get("text")) is not str:
                reject("unexpected_content_block")
            texts.append(block["text"])
            total_bytes += len(block["text"].encode("utf-8"))
            if total_bytes > MAX_RESPONSE_BYTES:
                reject("response_text_too_large")
        messages.append(texts)
    diagnostics["text_utf8_bytes"] = total_bytes
    if "commentary" in phases:
        if phases.count("final_answer") != 1 or phases[-1] != "final_answer" or None in phases:
            reject("ambiguous_final_phase")
        texts = messages[-1]
        diagnostics["ignored_commentary_messages"] = phases.count("commentary")
    elif len(messages) == 1:
        texts = messages[0]
    else:
        reject("ambiguous_final_message")
    if len(texts) != 1:
        reject("ambiguous_final_text")
    encoded = texts[0].encode("utf-8")
    diagnostics["final_text_sha256"] = hashlib.sha256(encoded).hexdigest()
    diagnostics["final_text_utf8_bytes"] = len(encoded)

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                reject("duplicate_json_key")
            value[key] = item
        return value

    def constant(_):
        reject("nonfinite_json_number")

    try:
        value = json.loads(texts[0], object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as error:
        diagnostics.update(json_error_line=error.lineno, json_error_column=error.colno,
                           json_error_position=error.pos)
        reject("invalid_json")
    except RecursionError:
        reject("json_nesting_too_deep")
    if type(value) is not dict:
        reject("final_json_not_object")
    diagnostics.update(code="decoded_object", final_object_field_count=len(value))
    return value
