#!/usr/bin/env python3
"""Recompute SP11 decisions from saved Gaussian inputs without its selector."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from scipy.stats import chi2

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/statistical_papers/sp11_gaussian_risk_20261010T170426_70c40563'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(scores: np.ndarray, level: float) -> tuple[np.ndarray,np.ndarray]:
    # A different computation: enumerate ascending candidate thresholds and
    # directly count both tails at each candidate, without prefix recursion.
    candidates=np.sort(np.abs(scores),axis=1)
    neg=(scores[:,None,:] <= -candidates[:,:,None]).sum(axis=2)
    pos=(scores[:,None,:] >= candidates[:,:,None]).sum(axis=2)
    passing=(1+neg)/(10+pos) <= level
    exists=passing.any(axis=1)
    first=passing.argmax(axis=1)
    false=np.where(exists,pos[np.arange(len(scores)),first],0)
    selected=exists | (level>=.1)
    return false/(10+false),selected


def main() -> None:
    start=time.perf_counter()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'results/statistical_papers'/('sp11_recheck_'+stamp)
    out.mkdir()
    shutil.copyfile(__file__,out/'executed_source.py')
    record={'status':'running','new_toroidal_selection_datasets':0,
            'new_gaussian_rows':0,'selector_or_generator_imported':False,
            'data_receipt_sha256':sha(DATA/'receipt.json'),'source_sha256':sha(Path(__file__))}
    try:
        frozen=json.loads((DATA/'receipt.json').read_text())
        for name,digest in frozen['file_sha256'].items():
            if sha(DATA/name)!=digest:
                raise ValueError('Changed frozen input: '+name)
        inputs=np.load(DATA/'raw_noise.npz')
        outcomes=np.load(DATA/'individual_results.npz')
        rows=list(csv.DictReader((DATA/'summary.csv').open()))
        recomputed={}
        largest=0.
        for row in rows:
            name=row['configuration']
            inflation=float(row['inflation'])
            rank=int(row['rank'])
            basis=inputs[{'common_positive':'common_basis','opposing':'opposing_basis','rank_three':'rank_three_basis'}[name]]
            scores=inputs['original_noise']+np.sqrt(inflation)*inputs['pilot_noise'][:,:rank]@basis.T
            fdp,selected=select(scores,float(row['input_q']))
            key=name+'_'+str(inflation)+'_'+row['method']
            np.testing.assert_array_equal(fdp,outcomes[key+'_fdp'])
            np.testing.assert_array_equal(selected,outcomes[key+'_selected'])
            for metric,value in [('fdr',fdp.mean()),('fdr_mcse',fdp.std(ddof=1)/np.sqrt(len(fdp))),
                                 ('power',selected.mean()),('power_mcse',selected.std(ddof=1)/np.sqrt(len(fdp))),
                                 ('empty',(~selected).mean())]:
                delta=abs(float(value)-float(row[metric]))
                if delta>1e-13:
                    raise ArithmeticError('Summary mismatch: '+metric)
                largest=max(largest,delta)
            crossing=rank*(1+inflation)*np.log1p(inflation)/inflation
            independent_tv=chi2.sf(crossing/(1+inflation),rank)-chi2.sf(crossing,rank)
            if abs(independent_tv-float(row['gaussian_tv_envelope']))>1e-12:
                raise ArithmeticError('Gaussian distance mismatch')
            recomputed[key]=(fdp,selected)
        pairs=list(csv.DictReader((DATA/'paired.csv').open()))
        for row in pairs:
            key=row['configuration']+'_'+str(float(row['inflation']))
            index=0 if row['metric']=='fdp' else 1
            difference=recomputed[key+'_budget_level'][index].astype(float)-recomputed[key+'_unadjusted'][index].astype(float)
            np.testing.assert_allclose([difference.mean(),difference.std(ddof=1)/np.sqrt(len(difference))],
                                       [float(row['budget_minus_unadjusted']),float(row['mcse'])],atol=1e-13,rtol=0.)
        record.update(status='complete',decisions_recomputed=3600000,
                      summary_rows=len(rows),paired_rows=len(pairs),maximum_summary_difference=largest)
    except BaseException as error:
        record.update(status='incomplete',error=repr(error))
        raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-start
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        print(out)


if __name__=='__main__':
    main()
