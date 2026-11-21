"""Support limits for each post-plan stability audit (Python 3.8+)."""


def parse_support_cap(value):
    """Return 'auto' or a nonnegative integer, including from CLI input."""
    if isinstance(value, str):
        if value == 'auto':
            return value
        try:
            value = int(value, 10)
        except ValueError:
            raise ValueError("support_cap must be 'auto' or a nonnegative integer") from None
    if type(value) is not int or value < 0:
        raise ValueError("support_cap must be 'auto' or a nonnegative integer")
    return value


def effective_support_cap(policy, remaining_count):
    """Return the maximum number of remaining parts eligible to be held."""
    if type(remaining_count) is not int or remaining_count < 0:
        raise ValueError('remaining_count must be a nonnegative integer')
    cap = parse_support_cap(policy)
    return remaining_count if cap == 'auto' else min(cap, remaining_count)
