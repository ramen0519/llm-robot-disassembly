"""Preview pinned knowledge tools with explicit local operations. API/physics = 0."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.knowledge_tools import KnowledgeCatalog
from orchestrator.lesson_memory import write_new
from worker.task_memory import _safe_path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    p.add_argument('--skill-library', required=True)
    p.add_argument('--lesson-library')
    p.add_argument('--memory')
    p.add_argument('--operations', help='Optional JSON array of explicit search/read requests. Not model-generated.')
    p.add_argument('--output', required=True)
    p.add_argument('--allow-fixture', action='store_true')
    args=p.parse_args(argv)
    def read(path):
        source=_safe_path(path)
        if source.stat().st_size > 32*1024*1024:
            raise ValueError('preview_input_byte_limit')
        return json.loads(source.read_text(encoding='utf-8'))
    # Check every path lexically BEFORE opening any file.
    from worker.task_memory import _lexical_path
    for path in (args.request,args.skill_library,args.lesson_library,args.memory,args.operations,args.output):
        if path is not None: _lexical_path(path)
    output=_safe_path(args.output)
    if output.exists(): raise ValueError('new_preview_output_required')
    catalog=KnowledgeCatalog(read(args.skill_library),read(args.lesson_library) if args.lesson_library else None,
        read(args.memory) if args.memory else None,allow_fixture=args.allow_fixture)
    session=catalog.session(read(args.request))
    operations=read(args.operations) if args.operations else []
    if type(operations) is not list or len(operations)>8: raise ValueError('preview_operation_limit')
    results=[session.execute(op) for op in operations]
    value=dict(kind='offline_knowledge_tool_preview', index=session.index(), results=results, view=session.view(),
               actual_api_calls=0, physics_calls=0, model_selected=False)
    write_new(output,value)
    print(json.dumps(dict(output=str(output), counts=value['index']['counts'],
        read_documents=len(session.read_documents),api_calls=0,physics_calls=0),ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
