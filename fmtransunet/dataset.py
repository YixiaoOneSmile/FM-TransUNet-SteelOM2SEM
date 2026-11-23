"""Export accepted same-field patches and construct field-group development folds."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from .registration import read_gray, warp

MATERIALS = ('spheroidized_annealed', 'quenched', 'tempered')
SIZE = 256


def grouped_splits(rows):
    identifiers = [r['crop_id'] for r in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate crop_id')
    groups = defaultdict(list)
    for r in rows:
        groups[r['group']].append(r)
    if any(len({r['material'] for r in rr}) != 1 for rr in groups.values()):
        raise ValueError('Each field group must be associated with one state')
    bins = [[] for _ in range(3)]
    for material in MATERIALS:
        candidates = [g for g, rr in groups.items() if rr[0]['material'] == material]
        if not candidates:
            continue
        if len(candidates) < 3:
            raise ValueError(f'{material} needs at least three groups for three folds')
        loads = [0, 0, 0]
        for group in sorted(candidates, key=lambda g: (-len(groups[g]), g)):
            index = min(range(3), key=lambda i: (loads[i], i))
            bins[index].append(group)
            loads[index] += len(groups[group])
    return [dict(id=f'fold_{i}', train=[r['crop_id'] for r in rows if r['group'] not in val],
                 val=[r['crop_id'] for r in rows if r['group'] in val], val_groups=sorted(val), test=[])
            for i, val in enumerate(bins)]


def sampling_weights(rows):
    counts = Counter(r['material'] for r in rows)
    groups = defaultdict(set)
    sizes = Counter(r['group'] for r in rows)
    for r in rows:
        groups[r['material']].add(r['group'])
    masses = {key: sqrt(n) for key, n in counts.items()}
    total = sum(masses.values())
    return [masses[r['material']] / total / len(groups[r['material']]) / sizes[r['group']] for r in rows]


def build(pairs_file, out):
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as stage:
        staged = Path(stage) / 'dataset'
        export(pairs_file, staged)
        staged.rename(out)


def export(pairs_file, out):
    pairs = json.loads(pairs_file.read_text())
    if not pairs:
        raise ValueError('At least one reviewed pair is required')
    checked = []
    source_groups = {}
    seen_pairs = set()
    for pair in pairs:
        key = str(pair['pair_id'])
        group, material = pair['group'], pair['material']
        pitch = float(pair['pitch_um'])
        if not key or not key.replace('-', '').replace('_', '').isalnum() or key in seen_pairs:
            raise ValueError(f'Invalid or repeated pair ID: {key}')
        seen_pairs.add(key)
        if not group or material not in MATERIALS or not isfinite(pitch) or pitch <= 0:
            raise ValueError(f'Invalid field group, state or pitch for {key}')
        if pair.get('accepted') is not True:
            raise ValueError(f'{key}: pair must be explicitly accepted after review')
        names = [pairs_file.parent / pair[k] for k in ('om', 'sem', 'valid_mask')]
        for source in names[:2]:
            digest = sha256(source.read_bytes()).hexdigest()
            previous = source_groups.setdefault(digest, group)
            if previous != group:
                raise ValueError(f'{source.name} is shared by groups {previous} and {group}')
        checked.append((pair, key, names, pitch))
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('om', 'sem'):
        (out / folder).mkdir()
    rows = []
    for pair, key, (om_path, sem_path, mask_path), pitch in checked:
        sem = read_gray(sem_path)
        if 'aligned_om' in pair:
            # Already reviewed in the native SEM grid, including any local refinement.
            om = read_gray(pairs_file.parent / pair['aligned_om'])
            support = np.full(sem.shape, 255, np.uint8)
        else:
            om, support = warp(read_gray(om_path), sem.shape, pair['matrix'])
        mask = read_gray(mask_path)
        if om.shape != sem.shape or mask.shape != sem.shape:
            raise ValueError(f'{key}: image or mask does not match the native SEM grid')
        for x, y in pair['crop_xy']:
            if not isinstance(x, int) or not isinstance(y, int) or min(x, y) < 0:
                raise ValueError(f'{key}: invalid patch origin')
            cut = (slice(y, y+SIZE), slice(x, x+SIZE))
            if sem[cut].shape != (SIZE, SIZE) or not np.all(mask[cut] > 0) or not np.all(support[cut] == 255):
                raise ValueError(f'{key}: patch {x},{y} crosses an excluded or unsupported region')
            crop = f'{key}_{x}_{y}'
            if (out / 'om' / f'{crop}.png').exists():
                raise ValueError(f'Duplicate crop: {crop}')
            for modality, tile in (('om', om[cut]), ('sem', sem[cut])):
                Image.fromarray(tile).save(out / modality / f'{crop}.png')
            rows.append(dict(crop_id=crop, group=pair['group'], material=pair['material'],
                             pitch_um=pitch, om=f'om/{crop}.png', sem=f'sem/{crop}.png'))
    folds = grouped_splits(rows)
    if any(not fold['train'] or not fold['val'] for fold in folds):
        raise ValueError('All three folds must contain training and validation data')
    (out / 'patches.json').write_text(json.dumps(rows, indent=2) + '\n')
    (out / 'folds.json').write_text(json.dumps(folds, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reviewed_pairs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.reviewed_pairs, args.output)


if __name__ == '__main__':
    main()
