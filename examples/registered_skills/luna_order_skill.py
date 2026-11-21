def propose(ctx):
    delta = ctx['component_delta']
    isolated = ctx['remainder_isolated_nodes']
    degree = ctx['contact_degree']
    volume = ctx['volume_fraction_if_valid']
    if delta is None or isolated is None or degree is None:
        return None
    if volume is None:
        volume = 0.0
    return -100.0 * delta - 10.0 * isolated - degree - 0.01 * volume