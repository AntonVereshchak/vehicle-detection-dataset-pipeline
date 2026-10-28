"""Create a reproducible chronological-block YOLO train/val split from verified_104.
Run in project root: python prepare_train_val.py
"""
from pathlib import Path
from collections import defaultdict
import csv
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'dataset' / 'verified_104'
DEST = ROOT / 'dataset' / 'vehicle_split_104'
VAL_COUNTS = {'A': 5, 'B': 4, 'C': 8, 'D': 0}
EXPECTED = {'A': 35, 'B': 20, 'C': 44, 'D': 5}

manifest = SOURCE / 'verified_manifest.csv'
if not manifest.is_file():
    sys.exit(f'Missing manifest: {manifest}')
with manifest.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
if len(rows) != 104:
    sys.exit(f'Expected 104 manifest entries, found {len(rows)}')

groups = defaultdict(list)
for row in rows:
    name = row['image']
    label = row['label']
    if not name.startswith('train_') or '_frame_' not in name:
        sys.exit(f'Unexpected image name: {name}')
    group = name.split('_frame_')[0].replace('train_', '')
    if group not in EXPECTED:
        sys.exit(f'Unexpected group: {group}')
    if Path(name).stem != Path(label).stem:
        sys.exit(f'Image/label mismatch: {name}, {label}')
    if not (SOURCE / 'images' / name).is_file() or not (SOURCE / 'labels' / label).is_file():
        sys.exit(f'Missing image or label for {name}')
    groups[group].append(row)

for group, expected in EXPECTED.items():
    if len(groups[group]) != expected:
        sys.exit(f'{group}: expected {expected}, got {len(groups[group])}')
if len({r['image'] for r in rows}) != 104:
    sys.exit('Duplicate image names in manifest')
if DEST.exists():
    sys.exit(f'Output already exists: {DEST}\nRename or remove it before rerunning to avoid stale files.')

splits = []
for group in 'ABCD':
    # Sort by original frame index, not by CVAT export order.
    ordered = sorted(groups[group], key=lambda r: int(Path(r['image']).stem.rsplit('_', 1)[-1]))
    n_val = VAL_COUNTS[group]
    # Reserve the final contiguous chronological block for validation.
    for idx, row in enumerate(ordered):
        split = 'val' if n_val and idx >= len(ordered) - n_val else 'train'
        splits.append((row, group, split))

for row, group, split in splits:
    for kind, filename in [('images', row['image']), ('labels', row['label'])]:
        target = DEST / kind / split / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / kind / filename, target)

with (DEST / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['group', 'split', 'image', 'label', 'objects', 'source_archive'])
    for row, group, split in splits:
        writer.writerow([group, split, row['image'], row['label'], row.get('objects', ''), row.get('source_archive', '')])

(DEST / 'data.yaml').write_text(
    f'path: {DEST.resolve().as_posix()}\n'
    'train: images/train\nval: images/val\n'
    'names:\n  0: Vehicle\n', encoding='utf-8'
)

train = sum(split == 'train' for _, _, split in splits)
val = sum(split == 'val' for _, _, split in splits)
print(f'SUCCESS: {train} train / {val} val images; {train + val} total.')
for group in 'ABCD':
    tr = sum(g == group and s == 'train' for _, g, s in splits)
    va = sum(g == group and s == 'val' for _, g, s in splits)
    print(f'  {group}: train={tr}, val={va}')
print(f'YAML: {DEST / "data.yaml"}')
print('CAUTION: val is a final chronological block from the SAME videos as train.')
print('This is preliminary within-video validation, NOT independent video-level generalization.')
print('There is NO temporal exclusion gap at the split boundary.')
