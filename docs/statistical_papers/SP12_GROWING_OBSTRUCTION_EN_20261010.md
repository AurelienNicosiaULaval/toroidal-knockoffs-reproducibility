# A growing-pair obstruction and its coupling scale

This proof concerns a specified sequence of known graphs of degree one. The signal fraction is one half and p/n tends to zero. It supplies a negative result, not a positive guarantee for sparse high-dimensional selection or learned graphs. The shared-sign mechanism has earlier generic precedents; the proposed contribution is its realization by this correctly specified model, estimator, regression and threshold.

## The model and statement

Fix g>0, kappa=g^2, sufficiently small for the existing one-pair drift lemma. A pair has X2 distributed as von Mises(kappa), an independent increment U distributed as von Mises(g), and X1=X2+U modulo 2 pi. The field kappa is known. Let K=K_n tend to infinity, m=m_n tend to infinity, and n/(K^3 m) tend to infinity. There are K independent identical pairs per subject. An independent pilot of m subjects estimates the common g from its Km increments by its nonnegative MLE. For selection subjects,

Y=sum_{j=1}^K cos(2 X_{2j-1}) + xi,

where xi is independent centered Gaussian noise with fixed positive variance. Generate witnesses from the kernel at the fitted g, using a fixed first-harmonic basis and fixed positive ridge penalty lambda. At a fixed level q in (0,1/4), fitted-law knockoff+ has FDR tending to 1/4, unconditional power tending to one and empty-selection probability tending to zero. The specified true-parameter kernel has exact FDR at most q at every n and power tending to one.

## Uniform regression calculation

For one pair, d_t is the eight-dimensional original/witness first-harmonic design; z=cos(2 X1). With the true g fixed and the working t near g, define mu_t=E d_t, G_t=Var d_t, b_t=G_t^{-1} Cov(d_t,z), and r_t=z-Ez-(d_t-mu_t)'b_t. Full conditional angular density makes G_g positive definite. Continuity and a compact interval about g give a common lower bound c>0 and bounded b_t. Pair independence gives the K-pair covariance Gamma_{K,t}=diag(G_t,...,G_t), coefficient b_{K,t}=(b_t,...,b_t), and residual R=sum r_{j,t}+xi. In particular ||b_{K,t}||<=C sqrt(K), E R^2<=C K and E R^4<=C K^2, uniformly in t. The r_{j,t} are bounded, independent and centered. All constants depend on the fixed one-pair model, not K,m,n.

Condition on a pilot t in that interval. Center at the population mean mu, put u_i=d_i-mu, and use bar u and bar R for sample means. Bounded features give

E ||n^{-1}sum u_i u_i' - Gamma_{K,t}||_F^2 <= C K^2/n,
E ||bar u||^2 <= C K/n,
E bar R^2 <= C K/n.

The empirical centered covariance Gamma_hat therefore differs in operator norm by O_P(K/sqrt(n)+K/n), which vanishes in the stated regime. Its inverse after adding lambda/n I is uniformly bounded with probability tending to one. The population normal equation E[uR]=0 and ||u||^2<=32K give

E ||n^{-1}sum u_i R_i||^2 <= C K^2/n.

The empirical centered residual score is gamma_hat=n^{-1}sum u_i R_i-bar u bar R=O_P(K/sqrt(n)+K/n). The exact centered ridge equation is

b_hat-b_{K,t}=(Gamma_hat+lambda/n I)^{-1}(gamma_hat-lambda/n b_{K,t}).

Thus ||b_hat-b_{K,t}||=O_P(K/sqrt(n)), uniformly near g. Because the pilot converges, removing the conditioning preserves this bound. A block norm difference is Lipschitz, so max_j |W_hat_j-w_j(t)|<=sqrt(2)||b_hat-b_{K,t}||. No fixed-dimensional covariance formula has been reused without a K-dependent bound.

## Pilot, null signs and the actual filter

The Km increments are iid with score cos U-A(g), variance A'(g)>0. The interior likelihood expansion gives sqrt(Km)(g_hat-g) converging to N(0,1/A'(g)); truncation is inactive with probability tending to one. The one-pair lemma gives a nonzero null derivative a<0, equal common null population blocks at g, and an active norm score w1(g)>0. Hence

max_{j<=K} |sqrt(Km) W_hat_{2j}-a sqrt(Km)(g_hat-g)| -> 0 in probability.

The regression contribution is O_P(sqrt(K^3 m/n)); the Taylor remainder is O_P((Km)^(-1/2)). Both vanish. The active scores are uniformly separated from zero and all absolute null scores. A continuous nondegenerate centered common limit implies that the K null scores are all positive or all negative, each with limiting probability one half; the mixed-sign event tends to zero.

On the all-positive event every one of the 2K scores is positive. The smallest positive threshold has ratio 1/(2K)<=q and selects all angles, with FDP=1/2. On the all-negative null event no null can be selected. The boundary has zero limiting probability and FDP is bounded, so FDR tends to 1/4. Independently of the null sign event, the threshold at the smallest active score has ratio 1/K<=q once K is large, and excludes negative nulls by separation. The smallest admissible threshold thus selects all active scores. This proves power tending to one and empty probability tending to zero. At the true parameter the same uniform regression calculation gives separation; exact known-law exchangeability and the sign-flip property give its finite-n FDR bound.

## What consistency does hold

Each conditional field changes by exactly (g_hat-g)v(neighbor), so the uniform field error over all angles and configurations is |g_hat-g|=O_P((Km)^(-1/2)). The Frobenius error of the stacked K interaction matrices is sqrt(2K)|g_hat-g|=O_P(m^(-1/2)); this rate is derived with K explicitly present. The per-subject covariate KL divergence is exactly

D_K=K{(g-g_hat)A(g)+log I0(g_hat)-log I0(g)}.

Taylor expansion gives D_K=K A'(g)(g_hat-g)^2/2+O_P(K|g_hat-g|^3)=O_P(1/m), and m D_K converges to chi-square(1)/2. Thus the law per subject and all its conditional fields are consistent. These facts do not assert closeness of the whole n-subject selection experiment; n/m diverges in this example. The general need to compare errors at the inference scale is established in knockoff robustness theory.

## An obstruction for any exact sign-flip reference vector

Let W* be any null score vector whose marginal law is invariant under independently reversing the K null signs, and couple it in any manner to the fitted scores. For any fixed c>=0,

liminf P{sqrt(Km) max_j |W_hat_{2j}-W*_{2j}|>c}
 >= 2 Phi(-c sqrt(A'(g))/|a|).

Indeed, if the maximum error is at most c/sqrt(Km), all fitted null scores have the same sign and their minimum absolute value exceeds that error, then all K reference null scores are nonzero and have the same sign. Exact sign flips bound the probability of the latter event by 2^(1-K), regardless of the magnitudes or their dependence. The fitted minimum scaled magnitude converges to |a Z|/sqrt(A'(g)). Subtract 2^(1-K) and the vanishing mixed-sign probability to obtain the inequality. Therefore any deterministic accuracy b_n with P(max error>b_n)->0 must satisfy b_n sqrt(Km)->infinity. Absolute consistency alone can still give b_n->0; it does not give relative accuracy near a shrinking null threshold.

This statement quantifies all couplings to an exact sign-flip reference, not merely a particular shared-random-number construction. It does not claim bibliographic priority for a general sign-flip obstruction.

## The noise scale for the specified oracle

At t=g, let l=(0,0,-1,0,0,0,1,0)', chi=l'G_g^{-1}(d_g-mu_g) and r=r_g. Put alpha=E chi^2 E r^2>0. Residual variance is positive because the second harmonic cannot be a linear combination of the first-harmonic original/witness features on their full angular support. The existing variance calculation gives

Var L_1=E(chi^2 r^2)+{(K-1)E r^2+Var xi}E chi^2,

so Var L_1/K tends to alpha, where L_1=chi_1(sum r_j+xi). E L_1^4<=C K^2. An independent-row triangular-array CLT yields n^{-1/2}sum L_{i,1}/sqrt(K) converging to N(0,alpha). The Gram inverse error times the residual score is O_P(K^2/n); after normalization sqrt(n/K), it is O_P(K^(3/2)/sqrt(n))=o_P(1). Centering, fixed ridge and the quadratic norm-score expansion have smaller or equal orders. Therefore sqrt(n/K) W^oracle_2 converges to N(0,alpha).

An accuracy sequence from the preceding obstruction has b_n/sqrt(K/n)->infinity. For this specified oracle, with G_n(t)=P(W^oracle_2>t) and t_n=sqrt(alpha K/n),

{G_n(t_n-b_n)-G_n(t_n+b_n)}/G_n(t_n)
 -> 1/{1-Phi(1)},

which does not vanish. This is the relative tail condition's obstruction for this oracle. The comparison is with ARK's arXiv version dated 4 June 2024, Conditions 1 and 5, rather than a claim that their theorem is false. Their other assumptions, including the sparse-null proportion used in their general theorem, are not asserted here. Fan et al.'s author version of 12 June 2026 similarly requires null symmetry/count concentration/localization; growing signals alone are not its sufficient conditions.

## Scope and numerical gate

The condition n/(K^3 m)->infinity is sufficient and deliberately coarse, not proved necessary. The result uses a known graph, a single correctly tied interaction and an omitted response harmonic. It does not handle general-matrix estimation, sparse-signal regimes, p>=n or learned graphs. The bounded nine-case SP12 protocol checks the moment calculations and uniform-regression implementation; it cannot numerically certify the FDR limit or the small-g existence premise. Retain any failures. No experiment is recorded as executed in this note before its actual run.
