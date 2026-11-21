"""HANDWRITTEN diagnostic fixture, never advertised as an LLM-generated skill."""
PROGRAM = {"language": "python_policy_v1", "source": """
def propose(ctx):
 l=ctx["last"]
 if l is None:return None
 f=l["feedback"]
 if f["path_success"]!=True or f["stability_success"]!=False:return None
 if not l["initial_disconnected"]:return None
 c=l["choice"]; s=[]
 for x in ctx["remaining_parts"]:
  if x!=c["part_id"] and x!=ctx["base_part"]:s.append(x)
 if not s or len(s)>ctx["max_grippers"]-1:return None
 q={"part_id":c["part_id"],"pose_id":c["pose_id"],"skill_id":c["skill_id"],
 "direction_id":c["direction_id"],"second_direction_id":c["second_direction_id"],
 "support_ids":s,"evidence_ids":[f["evidence_id"]]}
 if eligible(q):return q
 return None
"""}
