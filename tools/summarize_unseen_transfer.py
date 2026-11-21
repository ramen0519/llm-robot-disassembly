"""Read actual pilot artifacts; never infer missing metrics as success."""
from collections import Counter
from decimal import Decimal
import json,sys
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_unseen_transfer import RUN,read,write,verify_pin
from orchestrator.batch_budget import budget_summary

def optional(path):return read(path) if path.exists() else None

def summarize():
    verify_pin();spec=read(RUN/'protocol.json');rows=[]
    for item in spec['order']:
        ident,c=item['assembly'],item['condition'];p=RUN/'evaluations'/ident/c
        command=optional(RUN/'commands'/('evaluate_'+ident+'_'+c)/'command.json')
        row=dict(assembly=ident,condition=c,status='NOT_RUN',success=None,seconds=None,path_calls=None,
            stability_calls=None,max_depth=None,api_calls=0,estimated_usage_usd='0',skill_usage=None)
        if command:row.update(exit_code=command['exit_code'],seconds=command['elapsed_seconds'],status='ERROR')
        if c=='A':
            r=optional(p/'native_result.json')
            if r:row.update(status=r['status'],success=r['success'],max_depth=r['max_depth'],
                path_calls=r['budget']['counts']['path'],stability_calls=r['budget']['counts']['stability'],
                stop=r['native_stats'].get('stop_msg'),composite_attempts=len(r['attempts']),
                physics_seconds=r['budget']['physics_seconds'],physics_steps=r['budget']['physics_steps'])
        else:
            audit=optional(p/'audit.json');r=optional(p/'worker_result.json');meter=optional(p/'evaluation_budget.json')
            if audit:row.update(audit)
            if r:
                row.update(success=r['success'],max_depth=r.get('max_depth',r.get('stats',{}).get('max_depth')),
                    stop=r.get('stop_reason'),status=r['status'],composite_attempts=len(r['attempts']))
                if row['max_depth'] is None:
                    n=max((len(a['remaining_parts']) for a in r['attempts']),default=0)
                    row['max_depth']=max((n-len(a['remaining_parts'])+1 for a in r['attempts'] if a['accepted_for_transition']),default=0)
                row['failure_stages']=dict(Counter('path_failed' if a['raw'].get('path_success') is False else
                    'stability_failed' if a['raw'].get('stability_success') is False else a['status'] for a in r['attempts']))
                row['fallback_reasons']=dict(Counter(d['response']['fallback_reason'] for d in r['decisions'] if d['response']['fallback_reason']))
            if meter:row.update(path_calls=meter['counts']['path'],stability_calls=meter['counts']['stability'],
                physics_seconds=meter['physics_seconds'],physics_steps=meter['physics_steps'])
            before=optional(p/'budget_before.json');after=optional(p/'budget_after.json')
            if before and after:
                row.update(api_calls=after['reserved_calls']-before['reserved_calls'],
                    estimated_usage_usd=str(Decimal(after['usage_cost_usd'])-Decimal(before['usage_cost_usd'])))
            row['skill_usage']=optional(p/'skill_usage_summary.json')
            graph=optional(p/'graph_state.json')
            if graph and graph['status']=='ERROR':row['status']='ERROR'
        rows.append(row)
    report=dict(schema_version=1,rows=rows,api=budget_summary(RUN/'api_ledger.sqlite') if (RUN/'api_ledger.sqlite').exists() else None,
        pilot_only=True,independent_assemblies=3,seeds=[0],learning_during_evaluation=False,
        condition_definition=spec['conditions'],note='Native vs explicit is system-level; C vs D bundles Memory and Skill.')
    write(RUN/'comparison.json',report)
    lines=['# 미관측 조립품 12회 pilot — 실제 결과','',
        '| 조립품 | 조건 | 상태 | 전체 성공 | 최대 제거 깊이 | path | stability | 시간(s) | API | 사용량 추정($) |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        time='—' if r['seconds'] is None else str(round(r['seconds'],1))
        lines.append('| '+' | '.join(str(v) for v in [r['assembly'],r['condition'],r['status'],r['success'],
            r['max_depth'],r['path_calls'],r['stability_calls'],time,r['api_calls'],r['estimated_usage_usd']])+' |')
    lines+=['','A=원래 ASAP dfs+heur-out. B=동일정보 규칙+MS. C=Luna 장기지식 없음. D=Luna+동결 MS.',
        '실패/시간초과를 제품 교체나 재실행으로 지우지 않는다. UNKNOWN은 분해 불가능의 증명이 아니다.',
        '최대 제거 깊이는 하나의 실제 탐색 branch 진행량이며, 전체 성공/최종 계획과 구분한다.']
    with (RUN/'comparison.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'rows':len(rows),'api':report['api']},ensure_ascii=False))

if __name__=='__main__':summarize()
