# A restricted FDR consequence of the latent information budget

This note combines the project's conditional-redraw sensitivity bound with established Gaussian comparison and knockoff sign arguments. It is not a new general robustness framework or an exact fitted-law certificate. The close predecessor and the existing whitening construction must be credited in the article.

## Statement

Keep dimension, graph and harmonic basis fixed. Assume the regularity and independent-pilot regression limit of main Theorem 3, with m/n tending to a positive finite rho. At every null angle assume the common oracle original/witness coefficient block b_j is nonzero, so its norm-difference score is differentiable. At every alternative assume its oracle population score is strictly positive. Let the scalar null-score influences have positive diagonal oracle covariance Omega_s, let A_s be their working-parameter derivative, and assume a predetermined upper bound kappa_bar on the largest eigenvalue of I^(1/2) V I^(1/2). A correctly specified efficient likelihood pilot has kappa_bar=1.

Let N be the number of null angles, s=p-N, and r the rank of
E_rho = rho^(-1) Omega_s^(-1/2) A_s V A_s' Omega_s^(-1/2).
For deterministic input level q_0, the limiting FDR of the fitted-law norm-score filter is at most

q_0 + (N/p) delta_r(kappa_bar/(2 rho)).

Here delta_0(t)=delta_r(0)=0. For r>0 and t>0, with C_r the chi-square-r distribution function,

delta_r(t)=C_r(r(1+t)log(1+t)/t)-C_r(r log(1+t)/t).

The quantity delta_r is an exact upper envelope for the Gaussian total variation distance under rank r and operator inflation at most t. It is not a sharp FDR envelope. Without knowledge of the null set or sensitivity rank, replace N/p by one and r by min(p,k), where k is the fitted canonical parameter dimension. This remains an asymptotic conclusion under the stated response and score assumptions.

## Proof

A null norm-score influence is the unit direction of its common nonzero block applied to the original-minus-witness coefficient influence. It is odd under its own null swap, while all other null influences are unchanged. Their covariance is diagonal. Differentiating the zero oracle score along the true/kernel diagonal gives the cross-score identity. The conditional-redraw Schur argument therefore applies to these scalar influences and yields
A_s V A_s' <= (kappa_bar/2) Omega_s.
The two independent sample limits give a centered Gaussian null vector with covariance Omega_s + rho^(-1) A_s V A_s'. Standardization preserves each null sign. Apply the same inverse diagonal transformation when evaluating FDP under both reference laws: no rescaling of the implemented score order is proposed. Oracle standardized scores are independent fair-sign Gaussians; the fitted vector has covariance identity plus E_rho, whose nonzero eigenvalues lie in [0,t].

Rotate to its r-dimensional affected subspace, with eigenvalue inflations e_i. Start with the Gaussian pair N(0,I_r), N(0,(1+t)I_r). Apply independently in each coordinate the channel z_i -> sqrt(e_i/t) z_i + sqrt(1-e_i/t) epsilon_i. It sends the pair to N(0,I_r), N(0,diag(1+e_i)). Data processing bounds the resulting distance by that of the isotropic pair. Their densities cross at squared radius r(1+t)log(1+t)/t; integrating the density difference inside that ball gives the displayed chi-square difference. Equality is attained by isotropic inflation, for the distance criterion alone.

Positive population alternatives eventually exceed all absolute null scores. If any null is selected, all s alternatives are included. Hence the limiting FDP is a bounded function of the null sign and magnitude order, taking values between zero and N/p. Its expectation under fair signs is bounded by q_0 using the ordinary knockoff+ sign argument. A function in [0,N/p] changes expectation by at most (N/p) times total variation. This proves the bound. Zero and absolute-tie boundaries have Gaussian probability zero. The same conclusion holds for deterministic input levels tending to q_0: at fixed p, the attainable count ratios form a finite set, so subsequences reduce to one of the limiting threshold rules, whose fair-sign FDR is still at most q_0. No data-dependent choice of level is justified by this argument.

## Operational limitation

One can set a deterministic input equal to q minus the dimension-based envelope, or zero if negative, provided the covariance factor is bounded before looking at evaluated responses. This can be extremely conservative. If the positive input is below 1/p, knockoff+ returns no discoveries on every dataset, because (1+number of negatives)/(number of positives) is at least 1/p. Equality at 1/p is different: all p positive scores can pass. The information budget therefore does not provide useful calibration at every pilot ratio.

Zero common null blocks, invisible or nonpositive population alternatives, an unbounded pseudolikelihood efficiency factor, changing graphs and growing dimension require other arguments. An estimated covariance upper bound is not automatically a conservative bound. The Gaussian diagnostics in this step concern synthetic limiting score vectors, not additional toroidal selection datasets and not realizability of every covariance in the toroidal family.
