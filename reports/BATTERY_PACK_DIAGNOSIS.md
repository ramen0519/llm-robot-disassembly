# Battery-pack diagnostic summary — 2026-11-20 evidence

This summary reports an existing local run; no new simulation was executed during publication. Private lab CAD and raw geometry/trajectories are omitted.

The input had 13 CAD component occurrences / 14 solids, represented as 13 rigid parts. The case was fixed. Official GNN + DFS used six single-force directions, original pose, and up to two additional held parts. No BFS or programmed mid-path turn was connected.

The run attempted all 12 non-base first candidates: 72 path checks, 1 stability check, 0 accepted removals. Only the thin cover plate had a successful upward path. The completed search took about 265.73 seconds (worker about 273.24 seconds); this was not a search timeout.

A separate unchanged-physics diagnostic compared the intact assembly and the plate-removed assembly. Both failed at 20 steps with the same affected parts. Maximum measured translation was approximately 0.0207 mm, below the 2 mm displacement criterion. Failure came from loss of initially inferred contact, not demonstrated collapse.

Three holder/battery pairs have a CAD-axis gap of 1 mm. Under the chosen coordinate conversion the native contact threshold is also 1 mm. SDF distances were approximately 0.99996 mm initially and 1.001 mm after a small movement, crossing the contact threshold. This does not prove physical instability or impossibility of disassembly.

Future work: input/scale and surface-sampling checks, initial stability control, and a separately labeled contact/stability policy if required. No arbitrary threshold relaxation or simulator migration has been implemented in this snapshot.
