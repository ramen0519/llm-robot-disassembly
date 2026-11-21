"""Bounded, allowlisted exception evidence; never retain messages, URLs or keys."""
from copy import deepcopy
import errno
import http.client
import json
import re
import socket
import ssl
import urllib.error

QUOTA_CODES = frozenset(('insufficient_quota', 'billing_hard_limit_reached',
    'billing_not_active', 'organization_spend_limit_exceeded',
    'project_spend_limit_exceeded', 'organization_usage_limit_exceeded'))
RATE_CODES = frozenset(('rate_limit_exceeded', 'rate_limit_error', 'slow_down'))
PROVIDER_CODES = QUOTA_CODES | RATE_CODES | frozenset(('server_is_overloaded',
    'service_unavailable_error', 'invalid_api_key', 'model_not_found', 'invalid_request_error'))
_TYPES = (urllib.error.HTTPError, urllib.error.URLError, ssl.SSLCertVerificationError,
          ssl.SSLError, socket.gaierror, TimeoutError, ConnectionRefusedError,
          ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
          PermissionError, OSError, http.client.RemoteDisconnected,
          http.client.IncompleteRead, json.JSONDecodeError, ValueError, TypeError)


def _type(value):
    # Do not copy arbitrary user-defined class names into a diagnostic log.
    return next((cls.__name__ for cls in _TYPES if type(value) is cls), 'other_exception')


def _number(value, low=-100000, high=1000000):
    return value if type(value) is int and low <= value <= high else None


def _http_fields(error):
    cached = getattr(error, '_asap_http_diagnostics_v1', None)
    if cached is not None:
        return deepcopy(cached)
    result = dict(provider_error_code=None, provider_error_type=None,
                  provider_request_id=None, retry_after_seconds=None,
                  error_body_status='unavailable')
    headers = error.headers or {}
    request_id = headers.get('x-request-id')
    if type(request_id) is str and re.fullmatch(r'req_[0-9a-f]{16,64}', request_id):
        result['provider_request_id'] = request_id
    retry = headers.get('retry-after')
    if type(retry) is str and re.fullmatch(r'[0-9]{1,4}', retry) and int(retry) <= 3600:
        result['retry_after_seconds'] = int(retry)
    try:
        # Read only the existing error response, never issue another request.
        raw = error.read(8193)
        if raw:
            result['error_body_status'] = 'too_large' if len(raw) > 8192 else 'unrecognized'
            if len(raw) <= 8192:
                value = json.loads(raw)
                detail = value.get('error') if type(value) is dict else None
                if type(detail) is dict:
                    result['error_body_status'] = 'parsed_allowlist_only'
                    for source, target in (('code', 'provider_error_code'), ('type', 'provider_error_type')):
                        field = detail.get(source)
                        if type(field) is str and field in PROVIDER_CODES:
                            result[target] = field
    except Exception:
        result['error_body_status'] = 'unreadable_or_invalid'
    finally:
        try:
            error.close()
        except Exception:
            pass
    error._asap_http_diagnostics_v1 = deepcopy(result)
    return result


def diagnose(error):
    """Do not infer provider receipt/billing or a past error's missing cause."""
    root = error
    seen = set()
    for _ in range(4):
        if id(root) in seen or isinstance(root, urllib.error.HTTPError):
            break
        seen.add(id(root))
        reason = root.reason if isinstance(root, urllib.error.URLError) else None
        if not isinstance(reason, BaseException):
            break
        root = reason
    result = dict(schema_version=1, category='unknown_error', error_type=_type(error),
        root_error_type=_type(root), errno=_number(getattr(root, 'errno', None)),
        winerror=_number(getattr(root, 'winerror', None)), http_status=None,
        certificate_verify_code=None, automatic_retries=0, stop_paid_requests=False,
        provider_receipt='unknown', billing_status='unknown', raw_message_recorded=False)
    if isinstance(root, urllib.error.HTTPError):
        status = _number(root.code, 100, 599)
        result.update(_http_fields(root), http_status=status, category='http_error', stop_paid_requests=True)
        code, kind = result['provider_error_code'], result['provider_error_type']
        if status == 429:
            result['category'] = ('http_quota' if code in QUOTA_CODES
                                  else 'http_rate_limit' if code in RATE_CODES
                                  else 'http_quota' if kind in QUOTA_CODES
                                  else 'http_rate_limit' if kind in RATE_CODES
                                  else 'http_429_unspecified')
        elif status == 401: result['category'] = 'http_authentication'
        elif status == 403: result['category'] = 'http_permission'
        elif status is not None and status >= 500: result['category'] = 'http_server'
    elif isinstance(root, ssl.SSLCertVerificationError):
        result.update(category='tls_certificate', certificate_verify_code=_number(getattr(root, 'verify_code', None)))
    elif isinstance(root, ssl.SSLError): result['category'] = 'tls_error'
    elif isinstance(root, socket.gaierror): result['category'] = 'dns_error'
    elif isinstance(root, TimeoutError): result['category'] = 'transport_timeout'
    elif isinstance(root, ConnectionRefusedError): result['category'] = 'connection_refused'
    elif isinstance(root, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                           http.client.RemoteDisconnected, http.client.IncompleteRead)):
        result['category'] = 'connection_interrupted'
    elif isinstance(root, PermissionError): result['category'] = 'network_permission'
    elif isinstance(root, OSError):
        codes = {result['errno'], result['winerror']}
        if codes & {errno.ENETUNREACH, errno.EHOSTUNREACH, 10051, 10065}:
            result['category'] = 'network_unreachable'
        elif codes & {errno.ECONNREFUSED, 10061}: result['category'] = 'connection_refused'
        elif codes & {errno.ETIMEDOUT, 10060}: result['category'] = 'transport_timeout'
        elif codes & {errno.ECONNRESET, errno.ECONNABORTED, 10053, 10054}:
            result['category'] = 'connection_interrupted'
        else: result['category'] = 'network_os_error'
    elif isinstance(root, urllib.error.URLError): result['category'] = 'url_error_unspecified'
    # URLError is an OSError subclass with no errno when its reason is a string.
    if type(root) is urllib.error.URLError:
        result['category'] = 'url_error_unspecified'
    if isinstance(root, (OSError, http.client.HTTPException)) or isinstance(error, urllib.error.URLError):
        result['stop_paid_requests'] = True
    return result


class TransportCircuitOpen(ValueError):
    """A local rejection before a new network attempt or budget reservation."""
    pass
