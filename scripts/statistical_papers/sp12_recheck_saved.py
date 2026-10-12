"""Independently recheck saved SP12 diagnostics, without generating subjects.

QR ridge uses a different column order. Population integration uses Cartesian
original angles, cosine-only regression and unscaled Bessel functions. This is
an implementation check, not an FDR study or a numerical proof of an asymptotic
claim. No project generator, integrator or selector is imported.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from scipy.linalg import lstsq
from scipy.special import iv, roots_hermitenorm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'results/statistical_papers/sp12_growing_20261010T180723_6739aaf9'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cartesian_population(t: float) -> dict:
    """Integrate first-angle/root coordinates and conditional cosine moments."""
    nodes = 2 * np.pi * np.arange(40) / 40
    first, second = np.meshgrid(nodes, nodes, indexing='ij')
    first, second = first.ravel(), second.ravel()
    weight = np.exp(.04*np.cos(second)+.2*np.cos(first-second)) / (40**2*iv(0,.04)*iv(0,.2))
    original = np.column_stack([np.cos(first), np.cos(second)])
    vectors = np.column_stack([np.cos(first)+np.cos(second), np.sin(first)+np.sin(second)])
    response = np.cos(2*first)
    z, w = roots_hermitenorm(28)
    w = w / np.sqrt(2*np.pi)
    mu, raw, cross = np.zeros(4), np.zeros((4,4)), np.zeros(4)

    def moments(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radius = np.hypot(field[:,0], field[:,1])
        phase = np.arctan2(field[:,1], field[:,0])
        normalizer = iv(0,radius)
        return iv(1,radius)/normalizer*np.cos(phase), (1+iv(2,radius)/normalizer*np.cos(2*phase))/2

    for a, za in enumerate(z):
        for b, zb in enumerate(z):
            field = t*vectors+np.sqrt(t)*np.array([za,zb])
            mean1, second1 = moments(field)
            shifted = field+np.array([.04,0.])
            mean2, second2 = moments(shifted)
            conditional = np.column_stack([original,mean1,mean2])
            ww = weight*w[a]*w[b]
            mu += ww@conditional
            cross += (ww*response)@conditional
            raw += (conditional.T*ww)@conditional
            raw[2,2] += ww@(second1-mean1**2)
            raw[3,3] += ww@(second2-mean2**2)
    covariance = raw-np.outer(mu,mu)
    ez = weight@response
    cross -= mu*ez
    beta = np.linalg.solve(covariance,cross)
    var_r = weight@((response-ez)**2)-cross@beta
    l = np.array([0.,-1.,0.,1.])
    alpha = float(var_r*(l@np.linalg.solve(covariance,l)))
    return {'mu':mu,'gamma':covariance,'beta':beta,'alpha':alpha,'angular_mass':float(weight.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    args = parser.parse_args()
    reference = args.reference.resolve()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out = ROOT/'results/statistical_papers'/('sp12_recheck_'+stamp)
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    started = time.perf_counter()
    record = {'status':'running','reference':str(reference),'new_datasets':0,
              'source_sha256':sha(Path(__file__)),'scope':'Saved-data check, not an FDR calibration'}
    rows = []
    try:
        primary = json.loads((reference/'receipt.json').read_text())
        if primary['status']!='complete':
            raise ValueError('Primary diagnostics incomplete')
        for name,digest in primary['files_sha256'].items():
            if sha(reference/name)!=digest:
                raise AssertionError('Saved file changed: '+name)
        config = json.loads((reference/'configuration.json').read_text())
        if config['g']!=.2 or config['kappa']!=.04 or config['ridge']!=1. or config['level']!=.1:
            raise ValueError('Different experiment requires a separate checker')
        with (reference/'diagnostics.csv').open() as handle:
            diagnostics = {(int(r['K']),int(r['n'])):r for r in csv.DictReader(handle)}
        for case in config['cases']:
            k,n,m = case['K'],case['n'],case['m']
            name = f'K{k}_n{n}_m{m}'
            with np.load(reference/(name+'.npz'),allow_pickle=False) as saved:
                x1,root = saved['original_first'],saved['root']
                if np.max(np.abs(np.exp(1j*x1)-np.exp(1j*(root+saved['increment']))))>1e-13:
                    raise AssertionError('Original-angle identity')
                y = np.cos(2*x1).sum(axis=1)+saved['noise']
                np.testing.assert_allclose(y,saved['y'],rtol=0,atol=1e-13)
                t = float(saved['working_g'])
                mean = np.cos(saved['pilot_increment']).mean()
                if (t==0 and mean>0) or (t>0 and abs(iv(1,t)/iv(0,t)-mean)>1e-12):
                    raise AssertionError('Pilot likelihood equation')
                original = np.stack([x1,root],axis=2).reshape(n,2*k)
                witness = np.stack([saved['witness_first'],saved['witness_second']],axis=2).reshape(n,2*k)
                original_features = np.stack([np.cos(original),np.sin(original)],axis=2).reshape(n,4*k)
                witness_features = np.stack([np.cos(witness),np.sin(witness)],axis=2).reshape(n,4*k)
                d = np.concatenate([original_features,witness_features],axis=1)
                dc,yc = d-d.mean(axis=0),y-y.mean()
                augmented = np.vstack([dc,np.eye(8*k)])
                rhs = np.concatenate([yc,np.zeros(8*k)])
                b_global = lstsq(augmented,rhs,lapack_driver='gelsy')[0]
                indices = np.arange(8*k).reshape(k,8)
                order = np.concatenate([indices[:,:4].ravel(),indices[:,4:].ravel()])
                beta = np.empty(8*k); beta[order]=b_global
                beta_error = float(np.max(np.abs(beta-saved['beta'])))
                if beta_error>1e-10:
                    raise AssertionError('QR regression mismatch')
                wscore = np.linalg.norm(b_global[:4*k].reshape(2*k,2),axis=1)-np.linalg.norm(b_global[4*k:].reshape(2*k,2),axis=1)
                np.testing.assert_allclose(wscore,saved['scores'],rtol=0,atol=1e-10)
                chosen = np.zeros(2*k,dtype=bool)
                for threshold in sorted(set(abs(float(value)) for value in wscore if value!=0)):
                    positive = int(np.count_nonzero(wscore>=threshold))
                    negative = int(np.count_nonzero(wscore<=-threshold))
                    if 10*(1+negative)<=max(1,positive):
                        chosen = wscore>=threshold
                        break
                np.testing.assert_array_equal(chosen,saved['selected'])
                population = cartesian_population(t)
                cos_indices = [0,2,4,6]
                population_error = max(float(np.max(np.abs(population['beta']-saved['population_beta'][cos_indices]))),
                    float(np.max(np.abs(population['mu']-saved['mu'][cos_indices]))),
                    float(np.max(np.abs(population['gamma']-saved['gamma'][np.ix_(cos_indices,cos_indices)]))))
                if population_error>1e-9 or np.max(np.abs(saved['population_beta'][[1,3,5,7]]))>1e-10:
                    raise AssertionError('Independent population moments mismatch')
                target = np.tile(saved['population_beta'],k)
                norm_error = float(np.linalg.norm(beta-target))
                old = diagnostics[(k,n)]
                if abs(norm_error-float(old['regression_error']))>1e-10 or int(chosen.sum())!=int(old['selected']):
                    raise AssertionError('Recorded diagnostic mismatch')
                rows.append({**case,'qr_coefficient_max_error':beta_error,'population_max_error':population_error,
                             'selected':int(chosen.sum()),'new_datasets':0})
                print(name+' independently checked',flush=True)
        oracle = json.loads((reference/'oracle_population.json').read_text())
        own_oracle = cartesian_population(.2)
        if abs(own_oracle['alpha']-oracle['alpha'])>1e-9:
            raise AssertionError('Oracle noise scale')
        steps = [1e-4,5e-5]
        slopes = []
        for step in steps:
            plus,minus = cartesian_population(.2+step)['beta'],cartesian_population(.2-step)['beta']
            plus_score = abs(plus[1])-abs(plus[3])
            minus_score = abs(minus[1])-abs(minus[3])
            slopes.append(float((plus_score-minus_score)/(2*step)))
        if abs(slopes[-1]-oracle['negative_slope'])>1e-8 or slopes[-1]>=0:
            raise AssertionError('Independent null derivative')
        fisher = float(.5*(1+iv(2,.2)/iv(0,.2))-(iv(1,.2)/iv(0,.2))**2)
        scale = own_oracle['alpha']*fisher/slopes[-1]**2
        record.update(status='complete',saved_datasets_checked=len(rows),saved_selection_subjects_checked=126000,
                      max_qr_coefficient_error=max(r['qr_coefficient_max_error'] for r in rows),
                      max_population_moment_error=max(r['population_max_error'] for r in rows),
                      negative_null_slope=slopes[-1],oracle_alpha=own_oracle['alpha'],one_increment_information=fisher,
                      asymptotic_pilot_variance_to_oracle_ratio='n/(K^2 m)/scale_constant',
                      scale_constant=scale,
                      scope_of_scale='Leading K-growing Gaussian variance ratio at g=.2, not an FDR crossing or a sufficient finite-sample requirement',
                      all_nine_selections_empty=all(r['selected']==0 for r in rows))
    except Exception as error:
        record.update(status='incomplete',error=repr(error))
        raise
    finally:
        if rows:
            with (out/'checks.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        record['elapsed_seconds']=time.perf_counter()-started
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        print(out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
