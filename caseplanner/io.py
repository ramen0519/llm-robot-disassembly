"""Immutable run artifacts and location checks applied before filesystem access."""
import hashlib
import json
import os
import re
import stat
from pathlib import Path


def location(value):
    raw = str(value)
    if re.search(r'(^|[/\\])(?:e:|mnt[/\\]e)(?:[/\\]|$)', raw, re.I) or raw.lower().startswith(('e:', '\\\\', '//')):
        raise ValueError('disallowed volume')
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('parent traversal')
    path = path.absolute()
    for node in list(reversed(path.parents)) + [path]:
        if node.is_symlink():
            raise ValueError('linked path')
        if os.name == 'nt' and node.exists() and node.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('reparse path')
    return path


def sha(path):
    digest = hashlib.sha256()
    with location(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = location(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(location(path).read_text(encoding='utf-8-sig'))


def append(path, row):
    with location(path).open('a', encoding='utf8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
