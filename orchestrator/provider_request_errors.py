"""Safe structured errors before provider reservation or network transmission."""


class ProviderBodyError(ValueError):
    """A size guard rejection; never carry arbitrary exception/provider text."""

    def __init__(self, reason='plan_provider_body_too_large', *, body_bytes, max_bytes):
        if reason != 'plan_provider_body_too_large':
            raise ValueError('invalid_provider_body_error_code')
        if (type(body_bytes) is not int or type(max_bytes) is not int
                or body_bytes < 0 or max_bytes <= 0 or body_bytes <= max_bytes):
            raise ValueError('invalid_provider_body_error_sizes')
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = dict(stage='body_build', code=reason,
            body_utf8_bytes=body_bytes, maximum_utf8_bytes=max_bytes,
            excess_utf8_bytes=body_bytes-max_bytes, reservation_created=False,
            transmitted=False)


def local_request_diagnostics(error, stage):
    """Return only controlled categories; error messages may contain secrets."""
    if isinstance(error, ProviderBodyError):
        return dict(error.diagnostics)
    known_types = (ValueError, TypeError, RecursionError, OSError, ArithmeticError)
    category = next((cls.__name__ for cls in known_types if isinstance(error, cls)), 'Exception')
    return dict(stage=stage, code='provider_body_error' if stage == 'body_build'
                else 'cost_estimation_error', error_type=category,
                reservation_created=False, transmitted=False)
