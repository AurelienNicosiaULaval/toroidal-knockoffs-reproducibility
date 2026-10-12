# A simultaneous null-information budget

This result extends the scalar calculation of supplementary Proposition S5 to vector contrasts and an arbitrary fixed graph. It uses ordinary score differentiation, swap symmetry and the covariance projection inequality. These ingredients are inherited; the result is a specialization for the implemented block tests, not a new general information inequality.

Use the assumptions and notation of SP06.0. Write I=Var{T(X)} for the Fisher information of the identifiable canonical covariate family. Suppose I is positive definite. For each true null angle, L_j=C_j Γ^−1(D−ED)R, D_j=E[L_j L_jᵀ]>0 and B_j=−E[L_j{T(X)−ET(X)}ᵀ]. Stack the B_j into B over any set N of true null angles and set D_N=diag(D_j:j∈N). Then

\[
E[L_jL_k^T]=0\ (j\ne k,\ j,k\in N),\qquad
B^TD_N^{-1}B\preceq I.
\]

Consequently B I^−1 Bᵀ≤D_N. More generally, for any positive semidefinite pilot covariance V, put κ=λmax(I^(1/2) V I^(1/2)). Then

\[
BVB^T\preceq\kappa D_N,\qquad
D_N+\rho^{-1}BVB^T\preceq(1+\kappa/\rho)D_N.
\]

Proof. Let O_j swap the original and witness design blocks for null j. Oracle design exchangeability gives O_j Γ O_jᵀ=Γ. Null exchangeability including Y gives O_j β=β, so the centered regression residual R is invariant under this swap. Since C_j O_j=−C_j, L_j changes sign, while each L_k for k≠j is unchanged. Integrating the product establishes the zero cross-covariances; EL_j=0 follows also from the population normal equations. For U=D_N^−1/2 L_N and S=T−ET, Cov(U)=I and E[USᵀ]=−D_N^−1/2 B. The covariance of (U,S) is positive semidefinite; its Schur complement gives BᵀD_N^−1 B≤I. Whiten both sides by I^−1/2. The resulting matrix has operator norm at most one, which also gives B I^−1 Bᵀ≤D_N. Finally V≤κ I^−1 by the definition of κ. QED.

For an efficient independent likelihood pilot V=I^−1, κ=1. For compatible joint pseudolikelihood with subject loss score u, Hessian H and variability K, differentiating Eθ uθ=0 gives E[uSᵀ]=−H. The same covariance projection gives K≥H I^−1 H, hence V=H^−1 K H^−1≥I^−1 and κ≥1. This is the usual information loss of an estimating equation, not a new efficiency theorem. The bound quantifies the largest relative nuisance-error amplification; it does not order power or FDR across arbitrary couplings.

One conservative asymptotic comparator can avoid estimating B. Estimate I and V consistently from the independent pilot and D_j from the selection influence. Set νhat=1+(n/m)κhat and use qhat_jᵀ{νhat Dhat_j/n}^−1 qhat_j against χ²_(2H). Its null limit is stochastically dominated by that chi-square distribution, because the whitened limiting covariance has eigenvalues at most one. Its p-values are therefore asymptotically superuniform. The ordinary BY argument gives limsup FDR≤q|H0|/p. The finite-threshold summation bound is monotone in each null CDF: writing it as F(α_p)/p+Σ_(k<p)F(α_k){1/k−1/(k+1)} verifies this directly. Exact uniformity is not required for this conservative comparator.

This corollary retains p,H,graph fixed, a correctly specified regular pilot, m/n→ρ>0 or infinity, and positive-definite design and contrast covariances. I can be estimated as the empirical covariance of T(X) on the pilot, with no partition-function evaluation. It may be singular in a small pilot; then the proposed computation is unavailable and must not silently regularize away its hypothesis. Estimating the information costs O(m k²+k³) for k parameters. Its scalar factor is determined without selection responses, but the influence variance still uses them as justified studentization. The inequality is asymptotic; it is not a finite-sample certificate of the covariate fit and provides no guarantee under wrong conditionals or a learned growing graph. The conservative comparator need not have useful power and has not yet undergone a selection experiment.

Verification is a separate finite-state calculation, using exact SCIP recursion on three angles taking four values, a canonical pairwise chain with nonzero fields and anisotropic matrices, and a second-harmonic response with independent two-point noise. It evaluates all 8 192 augmented states, not a continuous von Mises simulation. The kernel derivative is checked independently by finite differences at a fixed true law; all null swaps, block orthogonality, information budget and covariance domination are checked. A duplicated whitened influence is a positive control showing why the orthogonality premise is necessary. The execution record, rather than this note alone, establishes which checks actually passed.
