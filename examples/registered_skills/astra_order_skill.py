def propose(ctx):
    shape = ctx.get("shortest_over_longest_extent")
    distance = ctx.get("distance_from_center_over_diagonal")
    volume = ctx.get("volume_fraction_if_valid")
    if shape is None:
        shape = 0.5
    if distance is None:
        distance = 0.0
    if volume is None:
        volume = 0.0
    shape = min(1.0, max(0.0, shape))
    distance = min(1.0, max(0.0, distance))
    volume = min(1.0, max(0.0, volume))
    geometry = shape * distance / (1.0 + volume)
    fragmentation = max(0, ctx["component_delta"])
    isolation = ctx["remainder_isolated_nodes"] / max(1, ctx["parent_count"] - 1)
    return round(geometry - 0.1 * fragmentation - 0.1 * isolation, 6)