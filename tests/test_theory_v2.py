"""Independent quadrature checks for the conditional-risk proposition."""
import numpy as np
from scipy.special import i0, i1


def test_conditional_kl_and_field_risk_bounds():
    angles=np.linspace(-np.pi,np.pi,32768,endpoint=False)
    v=np.c_[np.cos(angles),np.sin(angles)]
    for p,q in [(np.array([0.,0.]),np.array([.3,.8])),
                (np.array([1.2,-.4]),np.array([-.1,.7]))]:
        lp=v@p-np.log(2*np.pi*i0(np.linalg.norm(p)))
        lq=v@q-np.log(2*np.pi*i0(np.linalg.norm(q)))
        kl=np.mean(np.exp(lp)*(lp-lq))*2*np.pi
        norm=np.linalg.norm(p)
        mean=np.zeros(2) if norm==0 else i1(norm)/i0(norm)*p/norm
        bregman=np.log(i0(np.linalg.norm(q))/i0(norm))-mean@(q-p)
        error=np.sum((p-q)**2);K=max(norm,np.linalg.norm(q));c=np.exp(-2*K)/2
        assert abs(kl-bregman)<1e-13
        assert c*error/2 <= kl <= error/2


def test_sparse_signal_threshold_floor():
    from toroidal import select
    rng=np.random.default_rng(92389)
    for q in [.1,.2,.3]:
        for _ in range(200):
            w=rng.normal(size=40);w[:3]+=5
            chosen=select(w,q)
            assert len(chosen)==0 or len(chosen)>=int(np.ceil(1/q))
