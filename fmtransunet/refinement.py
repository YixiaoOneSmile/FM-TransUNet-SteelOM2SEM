"""Conservative local registration correction from independent texture tiles."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize

from .registration import read_gray, warp
RADIUS = 12
HALF = 48


def features(a):
    a = a.astype(np.float32)/255
    return [cv2.GaussianBlur(a,(0,0),s)-cv2.GaussianBlur(a,(0,0),3*s) for s in (2.,4.)]


def match(template_features, search_features, p, search_center):
    x,y = map(int,p)
    cx,cy = map(int,search_center)
    surfaces=[]
    for template,search in zip(template_features,search_features):
        patch=template[y-HALF:y+HALF,x-HALF:x+HALF]
        area=search[cy-HALF-RADIUS:cy+HALF+RADIUS,cx-HALF-RADIUS:cx+HALF+RADIUS]
        surfaces.append(np.abs(cv2.matchTemplate(area,patch,cv2.TM_CCOEFF_NORMED)))
    scores=np.mean(surfaces,axis=0)
    iy,ix=np.unravel_index(np.argmax(scores),scores.shape)
    peaks=[np.array(np.unravel_index(np.argmax(s),s.shape))[::-1] for s in surfaces]
    yy,xx=np.indices(scores.shape)
    outside=(xx-ix)**2+(yy-iy)**2>16
    return dict(delta=[int(ix-RADIUS),int(iy-RADIUS)],score=float(scores[iy,ix]),
                peak_margin=float(scores[iy,ix]-scores[outside].max()),
                scale_disagreement=float(np.linalg.norm(peaks[0]-peaks[1])),
                at_boundary=bool(ix in (0,2*RADIUS) or iy in (0,2*RADIUS)))


def design(points,kind):
    points=np.asarray(points,float)
    x=(points[:,0]-512)/512
    y=(points[:,1]-384)/512
    n=len(points)
    count={'identity':0,'translation':2,'affine':6,'affine_radial':7}[kind]
    a=np.zeros((n,2,count))
    if count:
        a[:,0,0]=1
        a[:,1,1]=1
    if count>=6:
        a[:,0,2]=x; a[:,0,3]=y
        a[:,1,4]=x; a[:,1,5]=y
    if count==7:
        a[:,0,6]=x*(x*x+y*y)
        a[:,1,6]=y*(x*x+y*y)
    return a


def predict(points,kind,params):
    return np.einsum('nij,j->ni',design(points,kind),np.asarray(params))


def fit(points, offsets, kind, grid):
    a=design(points,kind)
    count=a.shape[-1]
    bounds=np.array([8.,8.,6.,6.,6.,6.,4.][:count])
    def residual(p):
        error=(np.einsum('nij,j->ni',a,p)-offsets).ravel()
        return np.r_[error,.1*p[2:]]
    initial=least_squares(residual,np.zeros(count),bounds=(-bounds,bounds),loss='soft_l1',f_scale=1.5).x
    def slack(p):
        return 12.-np.linalg.norm(predict(grid,kind,p),axis=1)
    if slack(initial).min()>=0:
        return initial
    def objective(p):
        e=residual(p)/1.5
        return float(np.sum(2*1.5**2*(np.sqrt(1+e*e)-1)))
    result=minimize(objective,np.zeros(count),method='SLSQP',bounds=list(zip(-bounds,bounds)),
                    constraints=[{'type':'ineq','fun':slack}],options={'maxiter':300,'ftol':1e-8})
    if not result.success or slack(result.x).min() < -1e-5:
        raise RuntimeError('Bounded geometry fit failed: '+result.message)
    return result.x


def stats(values):
    return dict(median=float(np.median(values)),p90=float(np.percentile(values,90)),
                rmse=float(np.sqrt(np.mean(np.square(values)))))


def refine(om, sem, matrix, mask):
    """Fit bounded corrections on one set of local tiles, assess on held-out tiles."""
    if mask.shape != sem.shape:
        raise ValueError('Expected mask in the native SEM grid')
    aligned, support = warp(om, sem.shape, matrix)
    allowed = np.uint8((mask > 0) & (support > 0))
    safety = cv2.erode(allowed, np.ones((49, 49), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    of, sf = features(aligned), features(sem)
    height, width = sem.shape
    points = {'fit': [], 'check': []}
    margin = HALF + 2 * RADIUS
    for j, y in enumerate(range(margin, height-margin, 64)):
        for i, x in enumerate(range(margin, width-margin, 64)):
            if not safety[y-HALF:y+HALF, x-HALF:x+HALF].all():
                continue
            center = np.array([x, y])
            forward = match(sf, of, center, center)
            delta = np.asarray(forward['delta'])
            backward = match(of, sf, center+delta, center+delta)
            if (forward['score'] >= .2 and forward['peak_margin'] >= .015
                    and forward['scale_disagreement'] <= 3
                    and np.linalg.norm(delta+backward['delta']) <= 2
                    and not forward['at_boundary'] and not backward['at_boundary']):
                points['check' if (i+2*j) % 3 == 0 else 'fit'].append((center, delta))
    metadata = dict(model='identity', parameters=[], fit_tiles=len(points['fit']), check_tiles=len(points['check']))
    if len(points['fit']) < 4 or len(points['check']) < 3:
        return aligned, allowed * 255, metadata
    a, b = np.array(points['fit']).transpose(1, 0, 2)
    c, d = np.array(points['check']).transpose(1, 0, 2)
    baseline = stats(np.linalg.norm(d, axis=1))
    best = baseline['rmse']
    grid = np.array([[x, y] for y in np.linspace(0, height-1, 16) for x in np.linspace(0, width-1, 20)])
    for kind, minimum in [('translation', 4), ('affine', 8), ('affine_radial', 14)]:
        if len(a) < minimum:
            continue
        params = fit(a, b, kind, grid)
        trial = stats(np.linalg.norm(d-predict(c, kind, params), axis=1))
        if trial['rmse'] < .9*best and trial['median'] <= baseline['median'] and trial['p90'] <= baseline['p90']:
            best = trial['rmse']
            metadata.update(model=kind, parameters=params.tolist(), held_out_before=baseline, held_out_after=trial)
    if metadata['model'] == 'identity':
        return aligned, allowed * 255, metadata
    yy, xx = np.indices(sem.shape, dtype=float)
    spatial = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    coordinates = spatial + predict(spatial, metadata['model'], metadata['parameters'])
    field = coordinates.reshape(height, width, 2)
    jac = (np.gradient(field[..., 0], axis=1) * np.gradient(field[..., 1], axis=0)
           - np.gradient(field[..., 0], axis=0) * np.gradient(field[..., 1], axis=1))
    if not (jac.min() > .9 and jac.max() < 1.1):
        return aligned, allowed * 255, {**metadata, 'model': 'identity', 'parameters': [], 'reason': 'Jacobian bound'}
    homogeneous = np.column_stack((coordinates, np.ones(len(coordinates))))
    raw = np.einsum('ij,nj->ni', np.linalg.inv(matrix), homogeneous)
    positions = np.full_like(raw[:, :2], -1.)
    np.divide(raw[:, :2], raw[:, 2:], out=positions, where=np.abs(raw[:, 2:]) > 1e-9)
    maps = positions.reshape(height, width, 2).astype(np.float32)
    corrected = cv2.remap(om, maps, None, cv2.INTER_LINEAR)
    valid = (maps[..., 0] >= 0) & (maps[..., 0] < om.shape[1]-1) & (maps[..., 1] >= 0) & (maps[..., 1] < om.shape[0]-1)
    return corrected, np.uint8(valid & (allowed > 0)) * 255, metadata


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('om', 'sem', 'transform', 'valid_mask', 'output'):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    om, sem, mask = map(read_gray, (args.om, args.sem, args.valid_mask))
    matrix = np.asarray(json.loads(args.transform.read_text())['matrix'], dtype=float)
    corrected, valid, diagnostics = refine(om, sem, matrix, mask)
    args.output.mkdir(parents=True, exist_ok=False)
    for key, array in {'aligned_om': corrected, 'valid_mask': valid, 'overlay': np.dstack((sem, sem, corrected))}.items():
        if not cv2.imwrite(str(args.output / f'{key}.png'), array):
            raise OSError(f'Could not save {key}')
    (args.output / 'review.json').write_text(json.dumps(diagnostics, indent=2) + '\n')


if __name__ == '__main__':
    main()
