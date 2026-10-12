import itertools
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import check_grad
from toroidal import Model,EdgeSampler,vec,rot,knockstat,swap_log_ratio
from metro_circle import CircularMetro


def tiny():
    rng=np.random.default_rng(154)
    return Model(rng.normal(0,.2,(3,2)),np.array([[0,1],[0,2],[1,2]]),rng.normal(0,.3,(3,2,2)))


def test_local_logratio():
    P=tiny(); Q=Model(P.h+.04,P.edges,P.M*.8)
    rng=np.random.default_rng(3);x=rng.uniform(-np.pi,np.pi,(41,3));xt=EdgeSampler(Q).sample(rng,x)
    ell=swap_log_ratio(P,Q,x,xt)
    for j in range(3):
        y=x.copy();y[:,j]=xt[:,j]
        z=P.log_unnormalized(x)-Q.log_unnormalized(x)-P.log_unnormalized(y)+Q.log_unnormalized(y)
        np.testing.assert_allclose(z,ell[:,j],atol=2e-14)


def test_block_loading():
    P=tiny();sampler=EdgeSampler(P);A=np.zeros((6,6));D=np.zeros(3)
    for j,k,Bj,Bk in sampler.blocks:
        idx=[2*j,2*j+1,2*k,2*k+1];B=np.vstack([Bj,Bk]);A[np.ix_(idx,idx)]+=B@B.T
    for (j,k),M in zip(P.edges,P.M):D[[j,k]]+=np.linalg.norm(M,2)
    np.testing.assert_allclose(A,P.matrix()+np.diag(np.repeat(D,2)),atol=2e-14)


def test_sign_and_rotation():
    rng=np.random.default_rng(7);x=rng.uniform(-3,3,(100,3));xt=rng.uniform(-3,3,(100,3));y=rng.normal(size=100)
    w=knockstat(x,xt,y);a=rng.uniform(-3,3,3)
    np.testing.assert_allclose(w,knockstat(x+a,xt+a,y),atol=1e-13)
    x2=x.copy();t2=xt.copy();x2[:,1]=xt[:,1];t2[:,1]=x[:,1]
    want=w.copy();want[1]*=-1
    np.testing.assert_allclose(want,knockstat(x2,t2,y),atol=1e-13)


def test_metro_exhaustive_discrete():
    # Independent verification of the full sequential mechanism on 4^3 states.
    # Integrates every candidate and every acceptance pattern, including rejections.
    P=tiny();states=np.array(list(itertools.product(range(4),repeat=3)))
    theta=2*np.pi*states/4
    ix=np.repeat(np.arange(64),64);ip=np.tile(np.arange(64),64)
    X=theta[ix];C=theta[ip]
    lpx=P.log_unnormalized(theta);px=np.exp(lpx-logsumexp(lpx))
    for tau in (0.,1.5):
        sampler=CircularMetro(P,tau=tau,damping=.91)
        logq=np.sum(tau*np.cos(C-X),axis=1)-3*logsumexp(tau*np.cos(2*np.pi*np.arange(4)/4))
        joint=np.zeros((64,64));rng=np.random.default_rng(19)
        for pattern in itertools.product((False,True),repeat=3):
            forced=np.broadcast_to(pattern,X.shape)
            _,logbranch=sampler.sample(rng,X,proposals=C,forced=forced)
            xtstates=np.where(forced,states[ip],states[ix]);it=xtstates@np.array([16,4,1])
            np.add.at(joint,(ix,it),px[ix]*np.exp(logq+logbranch))
        np.testing.assert_allclose(joint.sum(1),px,atol=2e-14)
        for mask in range(1,8):
            a=states[ix].copy();b=states[ip].copy()
            for j in range(3):
                if mask&(1<<j):a[:,j],b[:,j]=states[ip,j],states[ix,j]
            aidx=a@np.array([16,4,1]);bidx=b@np.array([16,4,1])
            np.testing.assert_allclose(joint.ravel(),joint[aidx,bidx],atol=2e-14)


def test_frontier_cache_matches_full_recursion():
    from experimental_design import design,sample_design
    for name in ('chain','binary','star','cycle'):
        P=design(name,7);rng=np.random.default_rng(591)
        x=sample_design(rng,name,P,31);proposals=x+rng.vonmises(0,.8,x.shape)
        forced=rng.uniform(size=x.shape)<.5
        sparse=CircularMetro(P,tau=.8,damping=.97)
        dense=CircularMetro(P,tau=.8,damping=.97,order=sparse.order)
        a,la=sparse.sample(rng,x,proposals=proposals,forced=forced)
        b,lb=dense.sample(rng,x,proposals=proposals,forced=forced)
        np.testing.assert_allclose(a,b,atol=1e-14)
        np.testing.assert_allclose(la,lb,atol=2e-12)


def test_joint_pseudolikelihood_gradient():
    from fit_model import loss_grad
    rng=np.random.default_rng(161);P=tiny();x=rng.uniform(-np.pi,np.pi,(85,3));v=vec(x)
    par=np.r_[P.h.ravel(),P.M.ravel()]
    fun=lambda q:loss_grad(q,v,P.edges,.03)[0]
    jac=lambda q:loss_grad(q,v,P.edges,.03)[1]
    assert check_grad(fun,jac,par)<2e-5
    loss,g=loss_grad(np.zeros_like(par),v,P.edges)
    assert np.isfinite(loss) and np.all(np.isfinite(g))
    loss,g=loss_grad(par*1000,v,P.edges)
    assert np.isfinite(loss) and np.all(np.isfinite(g))
