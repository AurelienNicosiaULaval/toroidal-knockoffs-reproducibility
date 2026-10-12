# Local regularity of the operator-norm edge loading

This check closes a regularity question for the specific anisotropic star in SP06.1 and SP06.2. It does not extend SP06.0 to arbitrary learned graphs or to all matrix parameters, and it is not a novelty claim for matrix functional calculus.

Fix a finite graph. Suppose each true nonzero edge matrix has a simple largest singular value: σ1(M_e)>σ2(M_e)≥0. On a sufficiently small parameter neighborhood the largest singular value is smooth, positive, and separated from the other singular value. Hence s_e=||M_e||op is smooth. The augmentation matrix A_e=[[s_e I,M_e],[M_e^T,s_e I]] is smooth, positive semidefinite and of constant rank three. Its other eigenvalues are 2s_e, s_e+σ2 and s_e−σ2; all are bounded away from zero locally. Repeated positive eigenvalues, possible when σ2=0, cause no difficulty.

Let Π_e be the spectral projection onto its one-dimensional null space. The isolated zero eigenspace has a smooth projection locally, for instance by a resolvent contour separated from the positive eigenvalues. A_e+Π_e is positive definite. Its principal square root is smooth, and

\[
B_e=(A_e+\Pi_e)^{1/2}-\Pi_e
\]

is a smooth square root of A_e. Indeed A_e and Π_e commute, and the displayed expression vanishes on the null space and equals A_e^(1/2) on its orthogonal complement. Thus B_e B_e^T=A_e. The redundant fourth Gaussian coordinate is harmless. The implemented eigenfactor and this smooth factor induce exactly the same field distribution, so smoothness of a particular numerical eigenvector sign convention is unnecessary.

With fixed independent standard Gaussian vectors, the conditional field has a smooth representation as h+A v(x)+Σ_e B_e z_e. Its first two parameter derivatives are bounded locally by constants times 1+Σ_e||z_e||. Von Mises expectations of any bounded harmonic product have natural-field derivatives of every fixed order bounded uniformly: they are finite sums of cumulants of bounded trigonometric functions. The first two differentiated Gaussian integrands therefore have integrable polynomial envelopes. The same conclusion applies to design moments with Y or Y² by conditioning on X and using finite response moments; witness randomness is conditionally independent of Y. Dominated differentiation establishes the moment regularity used in SP06.0.

The SP06 star has singular values 0.8 and 0.5 on all eleven edges. It is in this neighborhood. Consistency places a regular fitted pilot in the neighborhood with probability tending to one. This check justifies the population moment derivatives for that particular model and the unchanged edge-loading kernel. Positive-definite design covariance, a regular pilot limit and a nonsingular contrast covariance remain separate hypotheses. Equal singular values, a true zero edge in a containing supergraph and graph changes are outside this argument. The rank of the Gaussian augmentation alone does not imply singularity of the angular design, whose von Mises conditional densities remain positive.
