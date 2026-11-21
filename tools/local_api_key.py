"""User-scoped Windows DPAPI storage. No API calls, key arguments, or key logs.

Register once in the key-bearing user's console; later launchers decrypt in
that same Windows account. No machine-wide decrypt flag or global env change.
"""
import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import getpass
import os
from pathlib import Path
import sys
import tempfile
import warnings

from run_luna_experiment import ROOT, safe

MAGIC = b'ASAP-OPENAI-DPAPI-v1\x00'
ENTROPY = b'disassembly_codex_ready_v2/openai_api_key/v1'
UI_FORBIDDEN = 1
MAX_BLOB = 65536


class KeyStoreError(RuntimeError):
    """Messages are fixed local codes, never secret or provider text."""


def _validate(key):
    if type(key) is not str or not 16 <= len(key) <= 4096 or any(not 33 <= ord(c) <= 126 for c in key):
        raise KeyStoreError('INVALID_KEY_FORMAT')
    return key


def _crypt(data: bytes, decrypt: bool) -> bytes:
    if sys.platform != 'win32':
        raise KeyStoreError('WINDOWS_DPAPI_REQUIRED')
    class Blob(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_ubyte))]
    crypt = ctypes.WinDLL('crypt32.dll', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32.dll', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    incoming = ctypes.create_string_buffer(data)
    entropy = ctypes.create_string_buffer(ENTROPY)
    source = Blob(len(data), ctypes.cast(incoming, ctypes.POINTER(ctypes.c_ubyte)))
    purpose = Blob(len(ENTROPY), ctypes.cast(entropy, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p if decrypt else wintypes.LPCWSTR,
                  ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    try:
        # LOCAL_MACHINE is intentionally absent; UI prompting is forbidden.
        ok = fn(ctypes.byref(source), None if decrypt else 'ASAP OpenAI API key',
                ctypes.byref(purpose), None, None, UI_FORBIDDEN, ctypes.byref(target))
        if not ok or not 0 < target.cbData <= MAX_BLOB:
            raise KeyStoreError('DPAPI_DECRYPT_FAILED_USE_REGISTERING_USER' if decrypt else 'DPAPI_ENCRYPT_FAILED')
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        ctypes.memset(incoming, 0, ctypes.sizeof(incoming))
        ctypes.memset(entropy, 0, ctypes.sizeof(entropy))
        if target.pbData:
            ctypes.memset(target.pbData, 0, target.cbData)
            kernel.LocalFree(target.pbData)


def key_path(root=ROOT):
    return safe(safe(root) / '.secrets/openai_api_key.dpapi')


def save_key(key, root=ROOT, *, replace=False):
    path = key_path(root)
    if not replace and path.exists():
        raise KeyStoreError('ALREADY_REGISTERED_NO_OVERWRITE')
    key = _validate(key)
    encrypted = _crypt(key.encode('ascii'), False)
    # Validate roundtrip before publishing anything; no plaintext file exists.
    if _crypt(encrypted, True).decode('ascii') != key:
        raise KeyStoreError('DPAPI_ROUNDTRIP_FAILED')
    path.parent.mkdir(exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.key-', delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(MAGIC + encrypted)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_path, path)
        else:
            # On Windows, rename fails if another process has already registered the key.
            try:
                os.rename(temporary_path, path)
            except FileExistsError:
                raise KeyStoreError('ALREADY_REGISTERED_NO_OVERWRITE') from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_key(root=ROOT):
    path = key_path(root)
    try:
        with path.open('rb') as stream:
            value = stream.read(MAX_BLOB + 1)
    except FileNotFoundError:
        return None
    if len(value) > MAX_BLOB or not value.startswith(MAGIC) or len(value) == len(MAGIC):
        raise KeyStoreError('INVALID_ENCRYPTED_KEY_FILE')
    try:
        return _validate(_crypt(value[len(MAGIC):], True).decode('ascii'))
    except UnicodeError:
        raise KeyStoreError('INVALID_DECRYPTED_KEY_FORMAT') from None


@contextmanager
def key_environment(root=ROOT):
    """Make a stored key available only for this process and its child execution."""
    previous = os.environ.get('OPENAI_API_KEY')
    if previous and previous.strip():
        yield 'process_environment'
        return
    key = load_key(root)
    if key is None:
        yield 'missing'
        return
    os.environ['OPENAI_API_KEY'] = key
    try:
        yield 'windows_dpapi'
    finally:
        if previous is None:
            os.environ.pop('OPENAI_API_KEY', None)
        else:
            os.environ['OPENAI_API_KEY'] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register', action='store_true')
    group.add_argument('--replace', action='store_true', help='Enter a new key at a hidden local console prompt')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.check:
            present = load_key() is not None
            print('STORED_KEY_READY' if present else 'NO_STORED_KEY')
            return 0 if present else 2
        if not args.replace and key_path().exists():
            print('ALREADY_REGISTERED: existing encrypted key was preserved.')
            return 3
        key = '' if args.replace else os.environ.get('OPENAI_API_KEY', '').strip()
        if not key:
            if not sys.stdin.isatty():
                print('INTERACTIVE_CONSOLE_REQUIRED: register from your PowerShell.')
                return 2
            with warnings.catch_warnings():
                warnings.simplefilter('error', getpass.GetPassWarning)
                key = getpass.getpass('OpenAI API key (hidden, not saved to shell history): ').strip()
        if args.replace:
            save_key(key, replace=True)
            status = 'REPLACED'
        else:
            save_key(key)
            status = 'REGISTERED'
        print(f'{status}: encrypted for this Windows user. API calls: 0.')
        return 0
    except KeyStoreError as error:
        print('KEY_STORE_ERROR=' + str(error))
        return 1
    except (Exception, KeyboardInterrupt) as error:
        print('KEY_STORE_ERROR_TYPE=' + type(error).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
