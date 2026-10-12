# Represented means on growing pair graphs

SP14, 10 October 2026. Mathematical argument written before the numerical checks. This is a specialization of conditional Gaussian regression and comparison of probability laws. Bibliographic priority is not certified. Numerical checks cannot prove the asymptotic statement.

## Statement

Fix g>0, known κ≥0, σ²>0 and λ>0. For each of K independent pairs per iid subject, independently draw X2j~VM(0,κ), Uj~VM(0,g), and set X2j−1=X2j+Uj modulo 2π. An independent pilot has m subjects and estimates the single tied g from its K m increments by the nonnegative MLE; κ stays fixed. The response is

\[
Y=\sum_{j=1}^K\cos(2X_{2j-1})+\xi,
\qquad \xi\sim N(0,\sigma^2),
\]

with noise independent of all covariates and witness randomness. Use the fitted Gaussian conditional-redraw kernel and centered ridge with H=2, fixed λ and any fixed q∈(0,1). If K,m→∞ and K²/n→0, then

\[
\limsup\operatorname{FDR}\leq q,\quad
\operatorname{Power}\to1,\quad
\Pr(\widehat S=\varnothing)\to0.
\]

No condition on m/n is imposed. This statement concerns a known degree-one graph, one learned scalar interaction, known fields, a represented response mean and Gaussian independent noise. Its rate is sufficient, not proved necessary. It does not cover general learned graphs, heteroskedasticity, response-adaptive bases or arbitrary binary responses.

## Conditional Gaussian transfer

Let d=16K and order the design by pair, with four features per angle (cos θ,sin θ,cos 2θ,sin 2θ). Let β0 concatenate the vector with value one on the original active cos-2 feature and zero elsewhere. Its norm is √K. With centered design Dc and S=DcᵀDc/n, the response is exactly Dcβ0 plus centered independent Gaussian noise. Witness generation uses no noise or evaluated response. Conditional on the full design and pilot,

\[
T_n:=\sqrt n(\widehat\beta-\beta_0)
\sim N(\mu_n,C_n),
\]

\[
\mu_n=-\frac\lambda{\sqrt n}(S+\lambda I/n)^{-1}\beta_0,
\quad C_n=\sigma^2(S+\lambda I/n)^{-1}S(S+\lambda I/n)^{-1}.
\tag{R1}
\]

Centering does not change this covariance: Dcᵀ1=0, so Dcᵀ times centered noise equals Dcᵀ times raw noise. The ridge factor is λ/n because the penalty in the original normal equation is λ, not nλ.

Write G_t for the centered covariance of one pair's H=2 design when originals have the true g and witnesses use t. At t=g, positive joint angular density and linear independence of the separate first/second harmonic functions imply G_g>0. Indeed a nonzero linear combination cannot be constant on the full four-angle support. On a compact interval J⊂(0,∞) around g, continuity gives cI≤G_t≤CI. The conditional von Mises moments are smooth in their natural fields. Differentiating F_t=t(v1+v2)+√t Z is dominated on J by C(1+||Z||). Bounded angular moments and Gaussian integrability therefore give ||G_t−G_g||F≤C|t−g|. κ is held fixed in this differentiation.

Conditional on the pilot, the K pairs are independent and the population covariance is ΓKt=diag(G_t,…,G_t). Bounded features give

\[
\|S-\Gamma_{K,\widehat g}\|_F
 =O_P(K/\sqrt n+K/n).
\tag{R2}
\]

For detail, the uncentered empirical covariance squared Frobenius error is at most CK²/n. Centering costs ||bar d−Ed||²=O_P(K/n). The constants are uniform for t∈J. The regular scalar increment MLE gives √(K m)(ĝ−g)=O_P(1); the probability of its boundary truncation vanishes. Hence

\[
\|\Gamma_{K,\widehat g}-\Gamma_{K,g}\|_F
 =\sqrt K\|G_{\widehat g}-G_g\|_F=O_P(m^{-1/2}),
\]

\[
\|S-\Gamma_{K,g}\|_F=O_P(K/\sqrt n+m^{-1/2})=o_P(1).
\tag{R3}
\]

The high-probability event of eigenvalues bounded between c/2 and 2C follows from the Frobenius bound. On that event, the inverse identity gives a dimension-independent Lipschitz bound in Frobenius norm. With ε=λ/n, scalar spectral calculus gives

\[
\|(S+\epsilon I)^{-1}S(S+\epsilon I)^{-1}-S^{-1}\|_F
\leq C\epsilon\sqrt{16K}.
\]

Consequently, writing C0=σ²ΓKg−1,

\[
\|C_n-C_0\|_F=O_P(K/\sqrt n+m^{-1/2}+\lambda\sqrt K/n)=o_P(1),
\quad \|\mu_n\|=O_P(\lambda\sqrt{K/n})=o_P(1).
\tag{R4}
\]

## Joint law, signs and the actual filter

Define E=C0−1/2 Cn C0−1/2−I. Since C0 has uniformly bounded positive eigenvalues, ||E||F=o_P(1). On ||E||op≤1/2 the Gaussian relative entropy is

\[
\operatorname{KL}\{N(\mu_n,C_n),N(0,C_0)\}
 =\tfrac12\{\mu_n^TC_0^{-1}\mu_n+
             \operatorname{tr}E-\log\det(I+E)\}
 \leq C(\|\mu_n\|^2+\|E\|_F^2)=o_P(1).
\tag{R5}
\]

This follows from x−log(1+x)≤Cx² for |x|≤1/2, applied to each eigenvalue. The variation distance is bounded by the square root of half this KL. On the complementary event it is at most one. Thus the conditional total variation distance tends to zero in probability, and its expectation tends to zero by boundedness. Mixture convexity transfers the bound to the unconditional Tn law. No continuity of the FDP function, no derivatives of norms at zero and no fixed-dimensional central limit theorem are needed here.

Let Z0~N(0,C0) and form ideal coefficients β0+Z0/√n, applying the same block-norm scores and actual finite knockoff+ rule. The known-law augmented design is invariant under every null-angle swap. Therefore ΓKg, its inverse C0 and the ideal Gaussian coefficient law are invariant under the corresponding swaps; β0 has zeros on both original and witness blocks of every null angle. The score law consequently has the exact null sign-flip property, including conditional on magnitudes and nonnull scores. The ordinary sign-flip bound gives ideal FDR≤q at each n.

For each n, FDP is the same bounded measurable function of the full coefficient vector under the fitted law and ideal law. The expectation difference is bounded by total variation even though the dimension grows and the filter is discontinuous. Hence fitted FDR≤q+o(1). This is an asymptotic proof, not a computable finite-sample certificate: G_g and its constants are unknown in an actual fitted-law analysis.

For power, conditionally the trace of Cn is O(K), and (R1) gives

\[
\|\widehat\beta-\beta_0\|=O_P(\sqrt{K/n})=o_P(1).
\tag{R6}
\]

This statement is unconditional because the bounded-spectrum event has probability tending to one. Block norm Lipschitz bounds make all active scores converge uniformly to one and all null scores uniformly to zero. Eventually a threshold at the smallest active score has ratio 1/K≤q, since every negative null has smaller magnitude. It is admissible and selects all K active coordinates. Power tends to one and empty probability to zero.

## Comparison and scope

For the same small g and κ=g² used by SP13, its regime n/(K²m)→∞ with K,m→∞ implies (R3). Thus, for the same pilot law and response, H=1 has FDR→1/4, whereas H=2 has limsup FDR≤q. This is a theorem about specified bases and means, not a universal recommendation to add harmonics. Existing represented-first-harmonic experiments retain a power cost from enrichment.

The Gaussian-law comparison is inherited, as is the use of probability-law distances in [BCS](https://arxiv.org/pdf/1801.03896) and [Zhang, Proposition 8](https://arxiv.org/pdf/2607.17084v3). [ARK](https://arxiv.org/pdf/2307.04400), Conditions 1 and 5, imposes accuracy conditions on a coupled score vector and its tail windows. We give a different sufficient comparison for this represented Gaussian response and pair design; no claim to improve the general ARK theorem or to establish bibliographic priority is made.

Proof written and reviewed internally. The Gaussian-noise condition, known fields, compatible tied scalar pilot, joint Frobenius rate and growing strong-signal count remain explicit. Numerical verification and human final review are separate requirements.
