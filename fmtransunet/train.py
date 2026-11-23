"""Train the supervised backbone or fine-tune FM-TransUNet on one field fold."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from pytorch_msssim import ssim
from torch.nn import functional as F

from .dataset import SIZE, grouped_splits, sampling_weights
from .model import TransUNetPlus
from .objectives import fft_magnitude_l1, gradient_loss_spatial, granulometry_loss, patch_spectrum_loss, VGG19PerceptualLoss


def load_images(root, rows):
    images = [[], []]
    for row in rows:
        for i, modality in enumerate(('om', 'sem')):
            with Image.open(root / row[modality]) as image:
                pixels = np.array(image.convert('L'))
            if pixels.shape != (SIZE, SIZE):
                raise ValueError(f'{row[modality]} is not a native 256×256 patch')
            images[i].append(torch.from_numpy(pixels).float().div(255).unsqueeze(0))
    return torch.stack(images[0]).repeat(1, 3, 1, 1), torch.stack(images[1])


def reconstruction(pred, target, perceptual):
    pixel = F.l1_loss(pred, target)
    structure = 1 - ssim(pred, target, data_range=1., size_average=True)
    spatial = gradient_loss_spatial(pred, target)
    vgg = 0. if perceptual is None else perceptual(pred, target, 1.)
    return pixel + .5 * structure + .05 * vgg + .2 * spatial


def auxiliary(pred, target, pitch, weights):
    terms = {'fft': fft_magnitude_l1, 'window': patch_spectrum_loss,
             'morphology': lambda a, b: granulometry_loss(a, b, pitch)}
    return {name: terms[name](pred, target) for name in weights}


def sampled_batches(weights, count, batch_size, schedule_seed, augment=True):
    gen = torch.Generator().manual_seed(schedule_seed)
    ids = torch.multinomial(weights, count * batch_size, replacement=True, generator=gen)
    if augment:
        codes = torch.randint(8, (count * batch_size,), generator=gen)
    else:
        codes = torch.zeros(count * batch_size, dtype=torch.long)
    return zip(ids.split(batch_size), codes.split(batch_size))


def batch(tensors, pitch, ids, codes, device):
    result = []
    for value in tensors:
        tiles = []
        for tile, code in zip(value[ids], codes.tolist()):
            tile = torch.rot90(tile, code % 4, (-2, -1))
            tiles.append(tile.flip(-1) if code >= 4 else tile)
        result.append(torch.stack(tiles).to(device))
    return *result, pitch[ids].to(device)


def freeze_for_fine_tuning(model):
    model.eval().requires_grad_(False)
    for layer in [*list(model.ups)[-2:], model.out_conv]:
        layer.requires_grad_(True)


def load_backbone(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    state = ckpt.get('model', ckpt)
    for prefix in ('foundation.backbone.', 'backbone.'):
        filtered = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
        if filtered:
            state = filtered
            break
    model.load_state_dict(state, strict=True)


def calibrate(model, perceptual, tensors, pitch, weights, device, requested):
    names = {'fft': 0., **dict.fromkeys(requested, 0.)}
    totals = defaultdict(float)
    gradients = []
    reference = next(model.ups[-1].parameters())
    for ids, codes in sampled_batches(weights, 4, 8, 4207000, augment=False):
        x, target, spacing = batch(tensors, pitch, ids, codes, device)
        pred = model(x)
        common = reconstruction(pred, target, perceptual)
        values = auxiliary(pred, target, spacing, names)
        g_common = torch.autograd.grad(common, reference, retain_graph=True)[0]
        g_fft = torch.autograd.grad(values.pop('fft'), reference, retain_graph=True)[0]
        gradients.append((g_common, g_fft))
        totals['common'] += g_common.norm().item()
        totals['fft'] += g_fft.norm().item()
        for name, value in values.items():
            totals[name] += torch.autograd.grad(value, reference, retain_graph=True)[0].norm().item()

    def ratio(numerator, denominator):
        return min(10., max(1e-8, .1 * numerator / max(denominator, 1e-12)))

    fft = ratio(totals['common'], totals['fft'])
    base_norm = sum((g0 + fft * gf).norm().item() for g0, gf in gradients)
    return {'fft': fft, **{k: ratio(base_norm, totals[k]) for k in requested}}


@torch.no_grad()
def validate(model, perceptual, tensors, rows, device, with_lpips=None):
    model.eval()
    by_field = defaultdict(list)
    for i, row in enumerate(rows):
        x = tensors[0][i:i+1].to(device)
        y = tensors[1][i:i+1].to(device)
        pred = model(x)
        values = {'common': reconstruction(pred, y, perceptual).item(),
                  'ssim': ssim(pred, y, data_range=1.).item(), 'mae': F.l1_loss(pred, y).item()}
        if with_lpips is not None:
            values['lpips'] = with_lpips(pred.repeat(1, 3, 1, 1)*2-1, y.repeat(1, 3, 1, 1)*2-1).item()
        by_field[(row['material'], row['group'])].append(values)
    by_state = defaultdict(list)
    for (state, _), records in by_field.items():
        by_state[state].append(mean_metrics(records))
    state_metrics = {state: mean_metrics(records) for state, records in by_state.items()}
    return {'state_balanced': mean_metrics(list(state_metrics.values())), 'by_state': state_metrics}


def mean_metrics(records):
    return {key: sum(r[key] for r in records) / len(records) for key in records[0]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('patches', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fold', type=int, choices=range(3), default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--parent', type=Path, help='Trusted same-fold backbone checkpoint for fine-tuning')
    p.add_argument('--variant', choices=('fft', 'window', 'morphology', 'fm'), default='fm')
    p.add_argument('--paper-config', type=Path, help='Use the published fold coefficients with configs/paper.json')
    p.add_argument('--smoke', action='store_true', help='Tiny network, no VGG/LPIPS download, one optimizer update')
    return p.parse_args()


def run(args):
    if args.paper_config and (not args.parent or args.smoke):
        raise ValueError('--paper-config requires a research-sized fine-tuning run with --parent')
    root = args.patches.parent
    rows = json.loads(args.patches.read_text())
    folds_path = root / 'folds.json'
    folds = json.loads(folds_path.read_text()) if folds_path.is_file() else grouped_splits(rows)
    mapping = {r['crop_id']: r for r in rows}
    split = folds[args.fold]
    train, val = ([mapping[k] for k in split[name]] for name in ('train', 'val'))
    if not train or not val or {r['group'] for r in train} & {r['group'] for r in val}:
        raise ValueError('Expected disjoint, nonempty training and validation field groups')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tx, vx = load_images(root, train), load_images(root, val)
    pitch = torch.tensor([r['pitch_um'] for r in train])
    weights = torch.tensor(sampling_weights(train), dtype=torch.double)
    shape = dict(channels=(4, 8, 16), embed_dim=16, heads=4, depth=1, feedforward=32) if args.smoke else {}
    # Retain the study backbone even though the joint model has no active conditioning embedding.
    model = TransUNetPlus(**shape).to(args.device)
    perceptual = None if args.smoke else VGG19PerceptualLoss().to(args.device).eval()
    evaluator = None
    if not args.smoke:
        import lpips
        evaluator = lpips.LPIPS(net='alex').to(args.device).eval().requires_grad_(False)
    coefficients = {}
    if args.parent:
        load_backbone(model, args.parent)
        freeze_for_fine_tuning(model)
        terms = {'fft': (), 'window': ('window',), 'morphology': ('morphology',),
                 'fm': ('window', 'morphology')}[args.variant]
        if args.paper_config:
            paper = json.loads(args.paper_config.read_text())
            weights_by_term = paper['weights_by_fold'][split['id']]
            coefficients = {name: weights_by_term[name] for name in ('fft', *terms)}
        elif args.smoke:
            coefficients = dict.fromkeys(('fft', *terms), .01)
        else:
            coefficients = calibrate(model, perceptual, tx, pitch, weights, args.device, terms)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-5 if args.parent else 5e-4, weight_decay=1e-4)
    epochs, steps, batch_size = (1, 1, 2) if args.smoke else ((50 if args.parent else 200), 24, 8)
    args.output.mkdir(parents=True, exist_ok=False)
    best = float('inf')
    for epoch in range(epochs):
        model.train(not bool(args.parent))
        if not args.parent:
            lr = 5e-4 * (epoch+1) / 10 if epoch < 10 else 1e-4 + 2e-4 * (1 + math.cos(math.pi * (epoch-10)/189))
            opt.param_groups[0]['lr'] = lr
        schedule_seed = args.seed*100000 + epoch + (7000 if args.parent else 0)
        for ids, codes in sampled_batches(weights, steps, batch_size, schedule_seed):
            x, target, spacing = batch(tx, pitch, ids, codes, args.device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = reconstruction(pred, target, perceptual)
            loss = loss + sum(coefficients[k] * v for k, v in auxiliary(pred, target, spacing, coefficients).items())
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            opt.step()
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch+1 == epochs:
            scores = validate(model, perceptual, vx, val, args.device, evaluator)
            summary = dict(epoch=epoch+1, coefficients=coefficients, fold=split['id'], seed=args.seed, **scores)
            print(json.dumps(summary), flush=True)
            payload = dict(model=model.state_dict(), metrics=summary)
            if not args.parent and scores['state_balanced']['common'] < best:
                best = scores['state_balanced']['common']
                torch.save(payload, args.output / 'best.pt')
    torch.save(payload, args.output / 'final.pt')


if __name__ == '__main__':
    run(parse_args())
