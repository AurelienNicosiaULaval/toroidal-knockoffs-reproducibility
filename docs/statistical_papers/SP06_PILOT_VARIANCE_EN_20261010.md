# Pilot-aware Wald contrasts: a fixed-dimensional result

Research note, version SP06.0. This is a different testing procedure from knockoff+. The derivation below and implementation checks do not constitute a finite-sample FDR experiment. The result uses classical two-step inference and Benjamini–Yekutieli multiple testing; its general ingredients are not claimed to be new.

## Setting and assumptions

Fix the number p of angles, a graph or fixed supergraph, and H harmonics. Let P_theta be an identifiable canonical exponential family on the torus with finite-dimensional parameter theta and bounded sufficient statistic T(X). The pairwise family with fields and compatible edge matrices is included. Its true parameter theta0 is interior. Let K_eta be a valid knockoff kernel for P_eta. Covariates of selection subjects have law P_theta0, and witnesses are generated from K_thetahat using a covariate-only pilot independent of the n selection subjects. Conditional on that pilot, all selection triples are iid.

The conditional law of Y given X is fixed when theta is varied. Under a null H0j it depends only on X_minus_j. Assume E|Y|^(2+delta)<infinity for some delta>0. The augmented design D consists of the original harmonic blocks followed by witness harmonic blocks, each of width d=2H. It is bounded. Its centered population covariance Gamma(theta,eta) is positive definite near (theta0,theta0). Population normal equations and their first two derivatives in theta and eta exist and are locally continuous; the conditional regression CLT and covariance LLNs hold locally in eta. These assumptions can be checked from a twice smoothly parameterized kernel with locally dominated derivatives. They do not automatically hold for nonsmooth spectral-norm loadings at every matrix parameter.

A concrete smooth version for general edge matrices is obtained by fixing strict scalar edge loadings c_e above the operator norms throughout a prespecified compact parameter neighborhood, using positive definite edge augmentation blocks and their smooth square roots. The ordinary positive-interaction chain is also smooth locally. This note does not assert differentiability for a learned graph or a loading selected from evaluated responses.

Assume sqrt(m)(thetahat−theta0) converges to N(0,V), and a covariate-only Vhat converges in probability to V. A compatible joint likelihood or pseudo-likelihood estimator can meet these assumptions at fixed graph under its separate regularity conditions. A product of incompatible nodewise regressions is not automatically such an estimator. Take m/n→rho in (0,infinity], retaining this root-m regularity assumption also when rho=infinity. Mere consistency at an unspecified rate is insufficient for the misspecified-response case. Use a deterministic ridge penalty lambda_n=o(sqrt(n)) on slopes, with an unpenalized intercept.

## Contrast and derivative

Let beta(theta,eta) be the centered population regression coefficient of Y on D under P_theta K_eta. For an angle j, C_j subtracts its witness coefficient block from its original coefficient block. Set q_j=C_j beta(theta0,theta0), let R=Y−EY−(D−ED)^T beta(theta0,theta0), and define the d-vector

\[
L_j=C_j\Gamma^{-1}(D-ED)R,
\qquad D_j=E[L_jL_j^T],
\qquad B_j=-E[L_j\{T(X)-ET(X)\}^T].
\]

Under H0j, q_j=0. This holds for a misspecified response regression as well: exchangeability of (X,witness,Y) under the null makes the design mean, covariance and normal equations invariant to the j swap. Uniqueness of beta then equates the two coefficient blocks. A nonlinear or binary response need not have a correctly represented conditional mean.

The kernel-parameter derivative of C_j beta(theta0,eta) at eta=theta0 equals B_j. To prove this, differentiate beta(theta,eta) with respect to theta while keeping eta fixed. Differentiating the centered normal equations gives Gamma^{-1} E[(D−ED)R s_theta(X)^T]. All terms from the mean derivatives vanish because ER=0 and E[(D−ED)R]=0. In a canonical family s_theta=T−ET. On the diagonal theta=eta, the null contrast is identically zero for the fixed conditional response law. The two derivatives therefore add to zero, proving the formula. No global normalizer is needed to estimate this cross-covariance.

## Null law and a computable variance

Theorem. Under the displayed assumptions and H0j, suppose S_j=D_j+rho^{-1}B_j V B_j^T is positive definite. Then

\[
\sqrt n\widehat q_j\Longrightarrow N(0,S_j).
\]

For rho=infinity use S_j=D_j and require it to be positive definite. Empirical centered regression residuals and the precision design give Lhat_ij. Define Dhat_j as their empirical covariance and Bhat_j as minus their empirical cross-covariance with T(X). Then

\[
\widehat\Sigma_j=\frac{\widehat D_j}{n}
+\frac{\widehat B_j\widehat V\widehat B_j^T}{m},
\qquad
\widehat q_j^T\widehat\Sigma_j^{-1}\widehat q_j
\Longrightarrow\chi_d^2.
\]

Proof. Conditional on the pilot, the regression CLT centers at beta(theta0,thetahat). Local continuity gives the oracle influence covariance D_j in the limit. The conditional characteristic functions converge in probability to the same nonrandom Gaussian characteristic function. Multiplying them by the bounded characteristic function of the independent pilot error proves joint convergence and limiting independence of the centered regression fluctuation and the pilot error. Taylor expansion of the population contrast gives B_j(thetahat−theta0); its quadratic remainder, multiplied by sqrt(n), is O_P(sqrt(n)/m)=o_P(1) when m/n→rho>0. The empirical regression linearization is uniform locally under the stated conditional LLN/CLT assumptions. The ridge contribution is O_P(lambda_n/n), also negligible at root n. Adding the independent terms proves the first display. For rho=infinity, the pilot term and remainder vanish. Bounded design and T, moment assumptions, conditional LLNs, and consistency of the population coefficients show Dhat_j→D_j and Bhat_j→B_j. Hence n Sigmahat_j→S_j. Slutsky's theorem and positive definiteness prove the chi-square limit. QED.

The estimate of B uses selection responses as part of studentization. It does not tune the covariate law or select a response-dependent pilot covariance. Its consistency is explicitly included in the proof. The statistic tests one angle with one d-dimensional test; its cosine and sine coordinates are not separate hypotheses.

If the true conditional mean is in the original harmonic linear span with a mean-zero conditional residual, the oracle coefficient is (beta_original,0), the residual has conditional mean zero given (X,witness), and B_j=0. This recovers the fixed-basis positive result rather than contradicting it. In the misspecified-response case the additional covariance need not vanish.

## Multiple testing and limits

Set P_j=Pr(chi_d^2 exceeds the displayed statistic) and apply the classical BY step-up threshold alpha_k=k q/(p sum_{l=1}^p 1/l). For fixed p, limsup FDR is at most q times the null fraction, hence at most q. No independence between the P_j is required.

For completeness, under selection a null P_j in (alpha_{k−1},alpha_k] requires at least k rejections, so its contribution to FDP is at most 1/k. Consequently FDR is bounded by the sum over null j of sum_k {F_j(alpha_k)−F_j(alpha_{k−1})}/k. Each null CDF converges at these finitely many thresholds to the uniform CDF by the theorem. The bound therefore converges to |H0| q/p. This is the usual BY argument, not a new multiple-testing theorem.

This procedure estimates pilot uncertainty rather than repairing finite-sample exchangeability. The guarantee is pointwise and asymptotic, at fixed p and fixed graph. The pilot must be correctly specified and regular; learned graphs, comparable high-dimensional samples, singular S_j, misspecified covariates and rho=0 are outside the theorem. A conditionally dependent alternative can have q_j=0 in the chosen harmonic projection, so the tests are not omnibus. A fixed alternative with q_j nonzero and the same covariance regularity is consistently detected. BY may be conservative. Adding the positive semidefinite pilot variance can only increase each computed p-value relative to the same working sample's unadjusted Wald value; this does not imply a universal FDR or power ordering.

The regression calculation costs O(n (4pH)^2+(4pH)^3), excluding pilot fitting and knockoff generation. Cross-covariances with k sufficient statistics add O(n pH k); all cross-covariances are computed from the same centered matrix products. Individual d by d inversions cost O(p d^3). No claim of the generator's linear complexity is extended to these response statistics.

## Direct precedents and residual contribution

| Prior result actually read | Overlap | Difference of this note |
| --- | --- | --- |
| Nguyen, Chevalier, Thirion and Arlot (2020), *Aggregation of Multiple Knockoffs*, definitions 4–5 and discussion, [primary PDF](https://proceedings.mlr.press/v119/nguyen20a/nguyen20a.pdf) | Knockoff-derived p-values and BY are already present. | Here the p-values are Wald block contrasts, with an explicit independent-pilot variance. No novelty claim for BY or knockoff-to-p-value conversion. |
| Niu, Chakraborty, Dukes and Katsevich (2024), *Reconciling model-X and doubly robust approaches to conditional independence testing*, [author version v2](https://arxiv.org/html/2211.14698v2), introduction and sections 2–4 passages | Learned covariate laws, Gaussian limits, response approximation and studentization are established issues. | The note retains first-order independent parametric pilot uncertainty instead of requiring it to disappear through accurate outcome residualization. This difference still needs a closer comparison before a novelty claim. |
| Katsevich and Ramdas (2022 arXiv v5), *On the power of conditional independence testing under model-X*, section 3.3, [author version](https://arxiv.org/html/2005.05506v5) | Vector-valued conditional independence tests and resampling-free asymptotic calibration. | Known first two conditional moments in MX(2) differ from an estimated finite-dimensional joint law with an explicit pilot covariance. Neither vector blocks nor normal calibration are new here. |
| Benjamini and Yekutieli (2001), *The control of the false discovery rate in multiple testing under dependency*, [DOI](https://doi.org/10.1214/aos/1013699998) | Arbitrary-dependence FDR via harmonic step-up correction. | The FDR step is inherited. DOI and formulation were cross-checked in Nguyen et al.; the tau.ac.il scanned preprint could not be reliably text-extracted during this pass. |
| Murphy and Topel (1985), *Estimation and Inference in Two-Step Econometric Models*, [official article page](https://www.tandfonline.com/doi/abs/10.1080/07350015.1985.10509471), abstract read | First-stage sampling error enters second-stage standard errors; treating the first fit as known is generally invalid. | Adding an estimated pilot-variance term is an instance of established two-step inference, not a new principle. Only the abstract was read in this pass. |
| Newey and McFadden (1994), *Large sample estimation and hypothesis testing*, [official chapter page](https://www.sciencedirect.com/science/article/abs/pii/S1573441205800054), abstract read; section 6 headings inventoried in a search result | The abstract explicitly covers two-step estimators and consistent asymptotic covariance estimation. | A general two-step calibration claim overlaps this framework. The chapter PDF fetch timed out; its detailed theorems were not read in this pass. The candidate specialization is the null-swap derivative identity and its toroidal block-test implementation. |

Status: an explicit fixed-dimensional derivation, plus algebraic implementation checks. No finite-sample simulation or practical robustness certificate has yet been produced for this new procedure. The direct two-step-inference precedents rule out novelty of the variance-addition principle. A closer audit of the swap-based specialization and its usefulness remains necessary. Neither a successful implementation test nor a sparse search establishes publishable originality.
