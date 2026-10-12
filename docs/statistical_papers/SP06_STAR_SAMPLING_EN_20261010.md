# Exact iid star simulation and the joint pilot sandwich

This note documents the model and algorithm used in SP06.1. It supplies a simulation identity, not a novelty claim for tree factorization or rejection sampling.

Let vertex 0 be the center, with arbitrary finite fields h_j and arbitrary real 2 by 2 matrices M_0j. The joint density is proportional to

\[
\exp\{h_0^Tv(x_0)+\sum_{j=1}^{p-1}h_j^Tv(x_j)
+\sum_{j=1}^{p-1}v(x_0)^TM_{0j}v(x_j)\}.
\]

Given x0, the leaves are independent von Mises with natural fields h_j+M_0j^T v(x0). Integrating them gives the root marginal proportional to exp(h0^T v(x0)) times

\[
W(x_0)=\prod_{j=1}^{p-1} I_0(\|h_j+M_{0j}^T v(x_0)\|).
\]

By the triangle inequality and monotonicity of I0 on nonnegative arguments,

\[
W(x_0)\leq C=\prod_{j=1}^{p-1}I_0(\|h_j\|+\|M_{0j}\|_{op}).
\]

Draw independent proposals from the root-field von Mises and accept each with probability W/C. The accepted density is exactly the root marginal by the ordinary rejection identity. Drawing its conditionally independent leaves completes an exact iid subject. Accepted proposals retained in order are iid; discarding unused accepted proposals from a vectorized batch does not change this law. No normalization of the full joint density, MCMC convergence assumption, or consecutive-chain approximation is used. Floating-point von Mises and Bessel evaluation retain their ordinary numerical approximation.

The expected proposal count per accepted root is C/E_proposal W, at most C because I0 is at least one. Evaluating each proposal costs O(p). This envelope can therefore be exponentially inefficient as the number or strength of edges grows. SP06.1 measures proposal acceptance and generation time instead of advertising a uniform linear exact-original simulator. The knockoff generator's own cost remains a separate calculation.

For the pilot, the canonical parameter order is all h coordinates followed by row-major M coordinates on the fixed edge list. The canonical sufficient statistics are all v(Xj) followed by row-major outer products v(X0) v(Xj)^T. With eta_j=h_j+sum_k M_jk v(Xk), the joint pseudo-loglikelihood per subject is

\[
\ell_i(\theta)=\sum_j\{\eta_j^T v(X_{ij})-\log(2\pi I_0(\|\eta_j\|))\}.
\]

Write r_ij=v(Xij)−E_theta[v(Xij)|X_i,minus_j]. Its field score is r_ij. The edge score is r_i0 v(Xij)^T+v(Xi0) r_ij^T. Thus a single matrix M_0j enters both node conditionals compatibly. The Hessian of the negative pseudo-loglikelihood is the sum of D_ij^T Cov_theta(v(Xij)|others) D_ij, where D_ij is the natural-field derivative matrix. This is the analytic Hessian already checked in the reference code.

SP06.1 minimizes the mean negative pseudo-loglikelihood plus ||theta||²/(2m). Positive penalty makes the finite-sample objective strictly convex and coercive. The inherited Newton polishing checks its stationary point. At fixed p and graph, the true-law support is the full torus and the natural-field linear maps are identifiable: a perturbation vanishing for every configuration must have every field and every edge-matrix coefficient zero. Conditional von Mises covariances are positive definite at finite natural fields. Consequently the population expected Hessian is positive definite. Under the usual fixed-dimensional regularity, the vanishing penalty has sqrt(m)/m→0 and does not change the centered first-order limit.

Let Hhat be the penalized mean Hessian and Khat the empirical covariance of the unpenalized joint subject scores. The implemented covariance of sqrt(m)(thetahat−theta0) is Vhat=Hhat^{-1} Khat Hhat^{-1}. Scores from different conditional nodes of one subject are not independent. Replacing Khat by a diagonal or by Hhat would omit those dependencies and is not the implemented procedure. Finite-sample shrinkage bias and sandwich error, especially at m=40, are not eliminated by this asymptotic formula.

Independent checks executed before the selection study compare the root integration to angular quadrature, the rejection envelope to a separate grid, root moments and conditional residuals to quadrature, joint scores to the reference gradient, the analytic Hessian to gradient finite differences, canonical parameter order to the energy, and the sandwich to direct inverse multiplication. The four tests pass. These checks validate these identities and their implementation; they are not evidence of FDR calibration or publishable originality.
