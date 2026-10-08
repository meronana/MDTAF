"""Build the data-page manifest from the curated datasets committed to the repository.

The curated files live in the repo itself (data/normalized_data/*_{source,target}.csv and
data/cv_datasets/), so the site links to them on GitHub. This script only writes
framwork/public/downloads/manifest.json (row counts, previews, checksums).
If the data are re-curated, re-run it and commit both the data and the manifest:
    python framwork/scripts/export_downloads.py
"""
import csv
import hashlib
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'data')
MANIFEST = os.path.join(ROOT, 'framwork', 'public', 'downloads', 'manifest.json')

# (endpoint, site key) — file names use the endpoint name as is
ENDPOINTS = [('fu', 'fu'), ('clearance', 'CL'), ('half_life', 't12')]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _short(c):
    # Preview only: long decimals to 4 places, long strings truncated (the files themselves stay original)
    try:
        if '.' in c:
            return f'{float(c):.4f}'
    except ValueError:
        pass
    return c if len(c) <= 60 else c[:57] + '…'


def csv_info(path, n_preview=5):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    # Preview keeps only the leading key columns even when there are many
    keep = min(len(header), 6)
    preview = [[_short(c) for c in r[:keep]] for r in body[:n_preview]]
    return {'rows': len(body), 'columns': header, 'preview_columns': header[:keep], 'preview': preview}


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, '/')


def main():
    files = []
    for ep, key in ENDPOINTS:
        for role in ('source', 'target'):
            path = os.path.join(DATA, 'normalized_data', f'{ep}_{role}.csv')
            files.append({'id': f'{ep}_{role}', 'endpoint': key, 'role': role, 'name': os.path.basename(path),
                          'path': rel(path), 'bytes': os.path.getsize(path), 'sha256': sha256(path),
                          **csv_info(path)})

    cv_dir = os.path.join(DATA, 'cv_datasets')
    cv_files = [os.path.join(b, n) for b, _, ns in os.walk(cv_dir) for n in ns if n.endswith('.csv')]
    with open(os.path.join(cv_dir, 'cv_summary.csv'), encoding='utf-8-sig') as f:
        cv_summary = list(csv.DictReader(f))

    manifest = {
        'files': files,
        'cv': {'path': rel(cv_dir), 'n_files': len(cv_files), 'bytes': sum(os.path.getsize(p) for p in cv_files)},
        'cv_summary': [{'fold': int(r['Fold']), 'endpoint': dict(ENDPOINTS)[r['Property']],
                        'source': int(r['Source']), 'train': int(r['Target_Train']),
                        'test': int(r['Target_Test'])} for r in cv_summary],
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f'wrote {MANIFEST}: {len(files)} files + {len(cv_files)} CV files')


if __name__ == '__main__':
    main()
