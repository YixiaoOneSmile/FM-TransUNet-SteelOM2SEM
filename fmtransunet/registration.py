"""Propose OM-to-SEM transforms and export geometry for visual review."""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    if image.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit microscopy image: {path}')
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def texture(image):
    """Difference of Gaussians from the coarse texture proposal."""
    gray = image.astype(np.float32) / 255
    return cv2.GaussianBlur(gray, (0, 0), 2.0) - cv2.GaussianBlur(gray, (0, 0), 6.0)


def propose(om, sem, scale):
    """Bounded rotation/scale and mirror search with free translation.

    scale is SEM pixels per OM pixel before rotation. A central template in
    reduced SEM supplies the matching signal. Confirm location in an overlay.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be finite and positive')
    h, w = sem.shape
    if min(h, w) < 64:
        raise ValueError('Coarse search needs at least 64 px per SEM side')
    sw, sh = max(32, round(w / 4)), max(32, round(h / 4))
    to_native = np.diag([w / sw, h / sh, 1.])
    tx, ty = sw // 4, sh // 6
    tw, th = sw - 2 * tx, sh - 2 * ty
    template = texture(cv2.resize(sem, (sw, sh), interpolation=cv2.INTER_AREA))[ty:ty+th, tx:tx+tw]
    if template.std() < 1e-6:
        raise ValueError('SEM template has insufficient texture')

    corners = np.array([[0, 0], [om.shape[1]-1, 0], [0, om.shape[0]-1],
                        [om.shape[1]-1, om.shape[0]-1]], dtype=float)
    best = None
    ones = np.ones_like(om)
    shrink = scale * min(sw / w, sh / h)
    sigma = max(0.0, 0.44 / shrink - 0.5)
    search_image = cv2.GaussianBlur(om, (0, 0), sigma) if sigma > 0 else om
    for mirror, ratio, angle in itertools.product((False, True), (.94, 1., 1.06), (-8., -4., 0., 4., 8.)):
        reflect = np.array([[-1., 0, om.shape[1]-1], [0, 1., 0], [0, 0, 1.]]) if mirror else np.eye(3)
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        a = scale * ratio
        transform = np.linalg.inv(to_native) @ np.array([[a*c, -a*s, 0], [a*s, a*c, 0], [0, 0, 1.]]) @ reflect
        xy = corners @ transform[:2, :2].T + transform[:2, 2]
        lo = np.floor(xy.min(axis=0))
        width, height = np.ceil(xy.max(axis=0) - lo + 1).astype(int)
        if width < tw or height < th or max(width, height) > 3000:
            continue
        transform[:2, 2] -= lo
        warped = cv2.warpPerspective(search_image, transform, (width, height))
        valid = cv2.warpPerspective(ones, transform, (width, height), flags=cv2.INTER_NEAREST)
        integral = cv2.integral(valid)
        area = integral[th:, tw:] - integral[:-th, tw:] - integral[th:, :-tw] + integral[:-th, :-tw]
        scores = np.abs(cv2.matchTemplate(texture(warped), template, cv2.TM_CCOEFF_NORMED))
        scores[area < th * tw] = -1
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score >= 0 and (best is None or score > best['score']):
            offset = np.array([[1., 0, tx-x], [0, 1., ty-y], [0, 0, 1.]])
            best = dict(score=score, angle_degrees=angle, reflected=mirror, matrix=(to_native @ offset @ transform).tolist())
    if best is None:
        raise ValueError('No fully supported candidate. Check scale or supply matching landmarks.')
    return best


def landmark_transform(om_points, sem_points):
    """Estimate a planar similarity or reflected similarity from ordered landmarks."""
    source, target = (np.asarray(p, dtype=float) for p in (om_points, sem_points))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError('Supply at least three paired (x, y) landmarks')
    x, y = source - source.mean(axis=0), target - target.mean(axis=0)
    if not np.isfinite(source).all() or not np.isfinite(target).all() or np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError('Landmarks must be finite and non-collinear')
    u, s, vh = np.linalg.svd(x.T @ y)
    linear = (u @ vh).T * s.sum() / np.square(x).sum()
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = target.mean(axis=0) - linear @ source.mean(axis=0)
    residuals = np.linalg.norm(source @ linear.T + matrix[:2, 2] - target, axis=1)
    return dict(matrix=matrix.tolist(), residuals_px=residuals.tolist(), reflected=bool(np.linalg.det(linear) < 0))


def warp(om, sem_shape, matrix):
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-10:
        raise ValueError('Expected an invertible 3×3 OM-to-SEM matrix')
    size = sem_shape[::-1]
    aligned = cv2.warpPerspective(om, transform, size)
    support = cv2.warpPerspective(np.full(om.shape, 255, np.uint8), transform, size, flags=cv2.INTER_NEAREST)
    return aligned, cv2.erode(support, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('om', type=Path)
    p.add_argument('sem', type=Path)
    p.add_argument('--output', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--scale', type=float)
    mode.add_argument('--landmarks', type=Path, help='JSON containing ordered om and sem coordinates')
    a = p.parse_args()
    om, sem = read_gray(a.om), read_gray(a.sem)
    if a.landmarks:
        points = json.loads(a.landmarks.read_text())
        result = landmark_transform(points['om'], points['sem'])
    else:
        result = propose(om, sem, a.scale)
    aligned, support = warp(om, sem.shape, result['matrix'])
    a.output.mkdir(parents=True, exist_ok=False)
    for name, image in {'aligned_om': aligned, 'support': support, 'overlay': np.dstack((sem, sem, aligned))}.items():
        if not cv2.imwrite(str(a.output / f'{name}.png'), image):
            raise OSError(f'Could not save {name}')
    result['status'] = 'proposal_requires_visual_review'
    (a.output / 'transform.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
