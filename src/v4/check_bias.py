"""Independent numerical checks of the analytic weak-interaction expansion."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from v3.population_pair import population


def main():
    rows=[];start=time.perf_counter()
    for g in [.1,.2,.4,1.]:
        k=g*g;step=g*1e-3
        for size,gh in [(48,16),(80,32)]:
            calculations=[population(gamma_true=g,gamma_work=t,h1=0.,h2=k,grid=size,hermite=gh) for t in [g-step,g,g+step]]
            slope=(calculations[2]['score'][1]-calculations[0]['score'][1])/(2*step)
            rows.append(dict(g=g,kappa=k,step=step,grid=size,hermite=gh,at_truth=calculations[1],
                lower_score=calculations[0]['score'],upper_score=calculations[2]['score'],
                slope=slope,leading_slope=-k*g/8,ratio_to_leading=slope/(-k*g/8)))
            print(g,size,slope,rows[-1]['ratio_to_leading'],flush=True)
    path=Path('results/v4/theory/bias_quadrature.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
