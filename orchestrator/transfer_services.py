"""Evaluation-only services: same inference contracts, bounded lossless packets."""
from copy import deepcopy
import json
from collections import Counter

from orchestrator.explicit_action_provider_v5 import project_request_v5,restore_request_v5
from orchestrator.skill_discovery import SkillDiscoveryService,SELECTION_INSTRUCTIONS,response_schema
from orchestrator.llm_ranker import _json

MAX_BODY_BYTES=100000


def pack_packet(value):
    """Readable, reversible tables and shared subobjects; never omit evidence."""
    counts=Counter()
    def count(v):
        if isinstance(v,(dict,list)):
            encoded=_json(v)
            if len(encoded)>=120:counts[encoded]+=1
            for x in (v.values() if isinstance(v,dict) else v):count(x)
        elif isinstance(v,str) and len(v)>=48:counts[_json(v)]+=1
    count(value)
    shared={};keys={}
    def pack(v,root=False):
        if isinstance(v,(dict,list)) or isinstance(v,str) and len(v)>=48:
            encoded=_json(v)
            if not root and counts[encoded]>1:
                if encoded not in keys:
                    key='v'+str(len(keys));keys[encoded]=key
                    shared[key]=pack(v,True)
                return {'$shared':keys[encoded]}
        if isinstance(v,dict):
            if any(k.startswith('$') for k in v):raise ValueError('reserved_packing_key')
            items=list(v.values())
            if len(items)>=2 and all(isinstance(x,dict) and set(x)==set(items[0]) for x in items):
                columns=sorted(items[0])
                return {'$keys':list(v),'$columns':columns,'$rows':[[pack(x[k]) for k in columns] for x in items]}
            return {k:pack(x) for k,x in v.items()}
        if isinstance(v,list):
            if len(v)>=2 and all(isinstance(x,dict) and set(x)==set(v[0]) for x in v):
                columns=sorted(v[0]);return {'$columns':columns,'$rows':[[pack(x[k]) for k in columns] for x in v]}
            return [pack(x) for x in v]
        return v
    data=pack(value,True)
    result={'encoding':'lossless_tables_v1','data':data,'shared_values':shared}
    if unpack_packet(result)!=value:raise ValueError('packet_roundtrip_failed')
    return result


def unpack_packet(packet):
    if packet['encoding']!='lossless_tables_v1':raise ValueError('unknown_packet_encoding')
    def unpack(v,stack=()):
        if isinstance(v,dict):
            if set(v)=={'$shared'}:
                key=v['$shared']
                if key in stack:raise ValueError('shared_cycle')
                return unpack(packet['shared_values'][key],stack+(key,))
            if set(v) in ({'$columns','$rows'},{'$keys','$columns','$rows'}):
                columns=v['$columns']
                if any(len(row)!=len(columns) for row in v['$rows']):raise ValueError('bad_table_width')
                rows=[{k:unpack(x,stack) for k,x in zip(columns,row)} for row in v['$rows']]
                if '$keys' in v:
                    if len(v['$keys'])!=len(rows):raise ValueError('bad_map_width')
                    return dict(zip(v['$keys'],rows))
                return rows
            return {k:unpack(x,stack) for k,x in v.items()}
        if isinstance(v,list):return [unpack(x,stack) for x in v]
        return v
    return unpack(packet['data'])


def compact_body(body):
    result=deepcopy(body)
    packet=json.loads(result['input'][0]['content'])
    result['input'][0]['content']=_json(pack_packet(packet))
    result['instructions']+=(' Input uses lossless_tables_v1: data is the original input. '
        'An object with $columns and $rows is a list of records: each row has the named columns in order. '
        'If $keys is also present, the records are dictionary values at those keys in order. '
        'An object with $shared points to the exact value in shared_values, expanded recursively. '
        'All facts, IDs, safety flags and action choices are preserved. This encoding is not a priority. '
        'Use the unchanged output JSON schema and original slot IDs.')
    return bounded_body(result)


def compact_selection(packet):
    if packet['operation']!='select':raise ValueError('evaluation_must_not_synthesize')
    result=deepcopy(packet)
    projection,audit=project_request_v5(packet['current_request'])
    result['current_request']=projection
    rows={r['evidence_id']:r for r in packet['current_request']['episode_feedback']+packet['current_request']['memory_references']}
    for row in packet['evidence']:
        if rows.get(row['evidence_id'])!=row:raise ValueError('selector_evidence_not_in_current_request')
    result['evidence']=[{'current_request_evidence_id':r['evidence_id']} for r in packet['evidence']]
    if restore_selection(result)!=packet:raise ValueError('selector_compaction_not_lossless')
    return result


def restore_selection(value):
    result=deepcopy(value)
    result['current_request']=restore_request_v5(result['current_request'])
    req=result['current_request']
    rows={r['evidence_id']:r for r in req['episode_feedback']+req['memory_references']}
    result['evidence']=[deepcopy(rows[r['current_request_evidence_id']]) for r in result['evidence']]
    return result


def bounded_body(body):
    if len(_json(body).encode('utf-8'))>MAX_BODY_BYTES:raise ValueError('evaluation_body_exceeds_100000_bytes')
    return body


def bound_actor(actor):
    original=actor.service._body
    actor.service._body=lambda envelope:compact_body(original(envelope))
    return actor


class TransferSkillSelector(SkillDiscoveryService):
    @staticmethod
    def _validate_envelope(envelope):
        if envelope.get('request',{}).get('operation')!='select':raise ValueError('evaluation_synthesis_forbidden')
        # Raw local evidence repeats source/request data. The transmitted packet
        # still has the strict 100 kB cap, before any ledger reservation.
        return SkillDiscoveryService._validate_envelope(envelope,max_packet_bytes=524288)

    def _body(self,envelope):
        if envelope['request']['operation']!='select':raise ValueError('evaluation_synthesis_forbidden')
        packet=envelope['request']
        body=dict(model=self.config['model'],store=False,instructions=SELECTION_INSTRUCTIONS,
            input=[dict(role='user',content=_json(compact_selection(packet)))],
            max_output_tokens=self.config['max_output_tokens'],reasoning={'effort':self.config['reasoning_effort']},
            text={'format':{'type':'json_schema','name':'generated_skill_decision','strict':True,'schema':response_schema(packet)}})
        body['instructions']+=' current_request is a lossless projection: request holds all current fields and shared_source_contexts expands source contexts by hash. evidence entries refer to exact evidence_ids in that request; no facts were removed. Source IDs never identify target parts.'
        return compact_body(body)
