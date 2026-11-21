#!/usr/bin/env python3
"""Validate this document distribution; NOT an ASAP/physics/program test."""
from pathlib import Path
import hashlib, json, re, sys
ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"SHA256SUMS.txt", "PACKAGE_VALIDATION.json"}

def main():
    errors=[]
    checksum_file=ROOT/"SHA256SUMS.txt"
    if not checksum_file.is_file():
        print("Missing SHA256SUMS.txt", file=sys.stderr); return 1
    checked=0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, rel=line.split("  ",1)
        path=(ROOT/rel).resolve()
        if ROOT not in path.parents:
            errors.append("Unsafe path: "+rel); continue
        if not path.is_file(): errors.append("Missing: "+rel); continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected: errors.append("Changed: "+rel)
        checked+=1
    for p in ROOT.rglob("*.md"):
        t=p.read_text(encoding="utf-8")
        if "\ufffd" in t: errors.append("Replacement character: "+str(p))
        fences=sum(1 for ln in t.splitlines() if ln.lstrip().startswith("```"))
        if fences % 2: errors.append("Unpaired code fence: "+str(p))
        # Only explicit local Markdown links; code/pseudocode paths are not required files.
        for link in re.findall(r"\]\(([^)]+)\)",t):
            if re.match(r"[a-zA-Z]+:",link) or link.startswith("#"): continue
            target=link.split("#",1)[0]
            if target and not (p.parent/target).exists(): errors.append("Broken link: "+str(p)+" -> "+link)
    for p in ROOT.rglob("*.json"):
        if p.name in EXCLUDE: continue
        try: json.loads(p.read_text(encoding="utf-8"))
        except (ValueError,UnicodeError) as e: errors.append("Invalid JSON: "+str(p)+": "+str(e))
    if (ROOT/"AGENTS.md").stat().st_size>=32768:
        errors.append("AGENTS.md exceeds 32 KiB distribution target")
    cfg=json.loads((ROOT/"configs/beam_baseline.example.json").read_text(encoding="utf-8"))
    if cfg["goal"]["mode"]!="native_asap": errors.append("Baseline goal changed")
    if cfg["extension"]["enabled"] or cfg["extension"]["allow_hard_pruning"]:
        errors.append("Baseline extensions unexpectedly enabled")
    meta=json.loads((ROOT/"PACK_METADATA.json").read_text(encoding="utf-8"))
    if meta["asap_executed_during_pack_creation"] is not False:
        errors.append("Incorrect execution claim")
    result={"kind":"DOCUMENT_DISTRIBUTION_VALIDATION_ONLY","passed":not errors,
            "checksummed_files":checked,"errors":errors,
            "asap_run":False,"research_tests_run":False}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
