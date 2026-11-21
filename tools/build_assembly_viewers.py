"""Insert measured geometry into the literal visualization template; no network."""
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    template = Path(__file__).with_name('assembly_viewer.template.html').read_text(encoding='utf-8')
    result = []
    for assembly_id in ['07009', '02628']:
        scene = data[assembly_id]
        html = template.replace('__ASSEMBLY_ID__', assembly_id).replace('__PART_COUNT__', str(len(scene['parts'])))
        html = html.replace('__SCENE_DATA__', json.dumps(scene, ensure_ascii=False, separators=(',', ':')))
        if len(html.encode()) > 1_000_000 or '__SCENE_DATA__' in html:
            raise ValueError('invalid_inline_visualization')
        path = args.destination / ('assembly-' + assembly_id + '.html')
        with path.open('x', encoding='utf-8') as stream:
            stream.write(html)
        content = path.read_bytes()
        result.append({'assembly': assembly_id, 'path': str(path), 'bytes': len(content),
                       'sha256': hashlib.sha256(content).hexdigest(), 'parts': len(scene['parts']),
                       'triangles': sum(p['triangle_count'] for p in scene['parts']),
                       'mesh_decimated': False, 'path_display_samples_per_step': 80 if 'steps' in scene else None})
    with args.manifest.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
