from pathlib import Path
import zipfile, csv, shutil, io, sys

# Put this script and verified_vehicle_labels_104.zip in your project root.
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'verified_vehicle_labels_104.zip'
SOURCE = ROOT / 'dataset' / 'selected'
OUTPUT = ROOT / 'dataset' / 'verified_104'

if not ARCHIVE.is_file():
    sys.exit(f'Missing archive: {ARCHIVE}')
if not SOURCE.is_dir():
    sys.exit(f'Missing selected images folder: {SOURCE}')

with zipfile.ZipFile(ARCHIVE) as z:
    rows = list(csv.DictReader(io.StringIO(z.read('verified_manifest.csv').decode('utf-8-sig'))))
    if len(rows) != 104:
        sys.exit(f'Unexpected manifest count: {len(rows)}')
    missing = []
    pairs = []
    for row in rows:
        name = row['image']
        group = name.split('_frame_')[0]  # train_A etc.
        source = SOURCE / group / name
        if not source.is_file():
            missing.append(str(source))
        pairs.append((source, name, row['label']))
    if missing:
        print('Missing original images (nothing copied):')
        print('\n'.join(missing[:25]))
        sys.exit(f'Missing {len(missing)} of 104 images. Check dataset/selected/train_A...train_D paths.')

    (OUTPUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'labels').mkdir(parents=True, exist_ok=True)
    for source, name, label in pairs:
        shutil.copy2(source, OUTPUT / 'images' / name)
        (OUTPUT / 'labels' / label).write_bytes(z.read('labels/' + label))
    (OUTPUT / 'data.yaml').write_text('path: ' + str(OUTPUT.resolve()).replace('\\', '/') + '\ntrain: images\nval: images\nnames:\n  0: Vehicle\n', encoding='utf-8')
    (OUTPUT / 'verified_manifest.csv').write_bytes(z.read('verified_manifest.csv'))

print('SUCCESS: 104 images and 104 labels copied to:', OUTPUT)
print('NOTE: data.yaml train and val currently point to the same images for structure checks ONLY.')
print('DO NOT evaluate model accuracy until a leakage-safe train/val split is created.')
