# A half-information budget for conditionally independent redraws

This refinement uses the conditional redraw structure of the implemented Gaussian augmentation. It is not a statement for every valid knockoff kernel. Its ingredients are score differentiation, swap symmetry, conditional variance and a covariance projection inequality. The result specializes those identities to shared independent-pilot error; bibliographic priority is not certified by the calculation.

## Setting

Use the fixed-dimensional assumptions of SP06.0, with bounded canonical sufficient statistic T(X), Fisher information I=Var(T)>0, and the null sensitivity identity B_j=−E[L_j(T−ET)ᵀ]. Fix any subset N of true null coordinates, let L_N stack their influences and let Ω_N=diag(E L_jL_jᵀ:j∈N)>0. The oracle kernel is a conditional independence redraw: a latent Z makes all original coordinates independent, and each witness is independently drawn from the same conditional coordinate law given Z. Thus originals and witnesses are mutually independent within and between coordinate pairs given Z. The law of (X,witness,Y) is invariant under each null swap. This last statement follows from the usual known-law model-X property and H0j.

Let X^N replace the original coordinates in N by their witnesses, leaving the other original coordinates fixed. Define

\[
 C_N=\operatorname{Var}\{E[T(X)\mid Z,X_{-N}]\},\qquad
 I_N^{\mathrm{res}}=E\operatorname{Var}\{T(X)\mid Z,X_{-N}\}=I-C_N.
\]

The latent variable may be continuous or singular. Only the stated conditional independence, moments and SP06 derivative assumptions are used.

## Theorem and proof

For B formed by stacking B_j over N,

\[
 B^T\Omega_N^{-1}B\preceq\tfrac12 I_N^{\mathrm{res}}\preceq\tfrac12 I.
\]

Consequently B I⁻¹Bᵀ≤Ω_N/2. For any pilot covariance V≥0 and κ=λmax(I^(1/2) V I^(1/2)),

\[
 BVB^T\preceq\tfrac\kappa2\Omega_N,\qquad
 \Omega_N+\rho^{-1}BVB^T\preceq(1+\kappa/(2\rho))\Omega_N.
\]

Proof. A null swap j changes L_j to −L_j and leaves every other L_k unchanged. The residual is invariant because the two null coefficient blocks coincide. Therefore Cov(L_N)=Ω_N. Swapping all coordinates in N simultaneously sends every L_j to −L_j and sends T(X) to T(X^N). Invariance of the triple gives E[L_N{T(X^N)−ET}ᵀ]=−E[L_N{T(X)−ET}ᵀ]=B. Hence, for Δ=T(X)−T(X^N), E[L_NΔᵀ]=−2B.

Conditional on (Z,X_-N), T(X) and T(X^N) are independent and identically distributed. This does not require conditioning on Y. Their covariance equals C_N, and their unconditional variances both equal I. Thus Var(Δ)=2(I−C_N)=2 I_N^res. The covariance matrix of (L_N,Δ) is positive semidefinite. Its Schur complement yields 4 BᵀΩ_N⁻¹B≤2 I_N^res, proving the first claim. Whiten by I⁻¹/²; equivalently the operator norm of Ω_N⁻¹/² B I⁻¹/² is at most 1/√2. This gives B I⁻¹Bᵀ≤Ω_N/2. Since V≤κI⁻¹, the two remaining inequalities follow. QED.

This proof does not infer FDR control from a variance bound. Off-diagonal pilot contributions can still correlate null signs. Greater latent predictability reduces the residual information bound but can reduce the response contrast's power; the bound is not a coupling-quality or power ordering. The graph, dimension, basis and pilot regime remain fixed, and misspecified conditional laws are not covered.

## Sharpness and a control outside the redraw class

Consider two angles taking four equally spaced values. Put t1=cos(X1) and z=cos(2X2), and consider the one-parameter canonical family p_θ(x) proportional to exp(θ t1 z). The true parameter is zero, and the response is Y=z. Coordinate 1 is conditionally null. Its first-harmonic projection is intentionally misspecified. The original law has full support on its finite grid, I=Var(t1 z)=1/2, and the augmented first-harmonic design has positive definite covariance in the examples below.

Use Z=z. Its two signs have equal prior probability because the conditional normalizers of X1 for the two signs coincide. Conditional on Z, draw original and witness X1 independently with mass proportional to exp(θ Z cos(x1)); draw original and witness X2 independently from the two angles with cos(2x2)=Z. This is a valid smooth conditional redraw kernel for p_θ. At θ=0, the design covariance is the identity times 1/2, the oracle coefficient is zero, L_1=2{v(X1)−v(witness1)}Y, Ω_1=4I2 and B_1=(−1,0)ᵀ. Thus B_1ᵀΩ_1⁻¹B_1=1/4=I/2. The constant 1/2 cannot be reduced uniformly over this bounded finite-angle conditional redraw class. This does not assert sharpness within the continuous bilinear first-harmonic family.

To show why the redraw restriction is needed, keep the same p_θ and conditional X2 redraw but replace the independent X1 pair by a symmetric Lancaster pair. Given Z, write q_θ for the tilted four-point law and f_θ(t)=(t−E_q t)/sqrt(Var_q t). The conditional pair density is q_θ(x1)q_θ(w1){1−γ f_θ(x1)f_θ(w1)}, with γ=1/4 and θ in a sufficiently small neighborhood of zero. It is nonnegative, has both q_θ marginals and is symmetric. It yields valid whole-coordinate swaps and a smooth kernel. At θ=0, its cosine copy correlation is −γ, its sine copy correlation is zero, and Ω_1=diag(4/(1+γ),4). The same null derivative B_1=(−1,0)ᵀ now gives BᵀΩ_1⁻¹B/I=(1+γ)/2=5/8>1/2. The general SP07 bound remains valid. This kernel is not a conditional independent redraw because a conditional independent pair cannot have this negative covariance.

The accompanying test enumerates the finite state space without project sampling or fitting code. It checks all four swap sets, positive definiteness, the information matrices, influence covariance, score derivative against a separate finite difference of population regression coefficients, equality in the redraw example and failure of the stronger bound in the valid negative-correlation control. These are exact finite-state sums with floating-point checks, not continuous von Mises simulations or FDR experiments.

## Consequence for the ten-pair calculation

Apply the theorem to the ten scalar smooth null block-norm influences of the independent-pair counterexample. Their variances equal τ², all derivatives equal a, and the efficient pooled pilot has V=I⁻¹. Then 10 a² V/τ²≤1/2, improving the general bound c=a²V/τ²≤1/10 to c≤1/20 for this particular redraw kernel.

The limiting Gaussian covariance has one inflated eigenvalue 1+10 c/ρ. For centered one-dimensional normals with variance 1+t versus 1, differentiating the density gives total variation at most φ(1) log(1+t)≤t/4. The other nine Gaussian directions coincide. The limiting FDP lies in [0,1/2], so

\[
 F_\rho\le F_0+\tfrac12\phi(1)\log(1+10c/\rho)
 \le F_0+1/(16\rho),\qquad
 F_0=467686735/5959489536.
\]

Since F0+1/48<1/10, ρ≥3 suffices for asymptotic FDR≤0.1 in that counterexample. The previous ratio six is still a valid historical sufficient condition but is no longer the strongest bound derived in this project. Neither condition is necessary or a finite-sample pilot-size certificate. The unit-parameter experiments remain outside the analytic small-interaction premises.
