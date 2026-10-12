# Loading order for Gaussian toroidal redraws

This note concerns a known, fixed covariate law. It uses classical Gaussian conditioning, comparison of experiments and conditional-expectation covariance ordering. It does not establish a new general principle of data augmentation, an ordering of power, or a fitted-law FDR guarantee. The numerical checks were executed separately under the frozen protocol; their receipt identifies the scope and outputs below.

## Setting and exact comparison

Fix the finite-field bilinear torus density with off-diagonal block matrix J, and write s(x)=(v(x_1)^T,...,v(x_p)^T)^T. Let D=diag(d_j I_2), with d_j>=0 and A=J+D>=0. Define G_A|X=x~N(A s(x),A), F_A=h+G_A, and redraw all witness coordinates independently from VM(F_Aj). The original conditional law given F_A is the same product law. Denote the reversible redraw operator by K_A on L^2(P).

Theorem (exact loading characterization). For two such admissible loadings D_1,D_2, the following are equivalent:

1. d_2j>=d_1j for every j.
2. <f,K_A2 f>_P>=<f,K_A1 f>_P for every f in L^2(P).

Under these conditions the latent prediction risks E Var(f(X)|F_A2) are no larger than those under F_A1. For any square-integrable vector statistic T and any set N, the conditional residual information E Var(T(X)|F_A2,X_-N) is also no larger in positive-semidefinite order. These conclusions concern the latent variables and one-step covariance; they do not compare optimal reconstruction from the observed witness vector.

Proof of sufficiency, including singular A_1. Put R=D_2-D_1>=0. Given X, generate G_1~N(A_1s(X),A_1) and an independent G_R~N(Rs(X),R), and set G_2=G_1+G_R. The latter has the A_2 channel law. Since range(A_1) is contained in range(A_2), Gaussian conditioning gives

G_1|G_2,X ~ N(A_1 A_2^+ G_2, A_1-A_1 A_2^+ A_1).

The mean and covariance do not depend on X. Thus X -> G_2 -> G_1 is a Markov chain: the lower-loading channel is a random transformation of the higher-loading one. This is the familiar comparison-of-experiments argument. With m_a=E[f(X)|G_a], the tower property gives m_1=E[m_2|G_1]. Hence

E(m_2^2)-E(m_1^2)=E(m_2-m_1)^2>=0.

Conditional independence of original and witness given G_a implies <f,K_Aa f>=E(m_a^2). The same tower argument, conditioning additionally on X_-N and using vector conditional covariance, proves the residual-information comparison. No inverse of A_1 is required. Its Moore-Penrose bridge covariance is positive semidefinite because it is a Gaussian conditional covariance.

For necessity, the marginal joint law of (X,F_Aj) depends on D only through d_j: F_Aj|X=x~N(h_j+d_jv(x_j)+sum_k M_jk v(x_k),d_j I_2). Furthermore, X_j|F_A is VM(F_Aj), so the conditional expectation of any function of X_j depends on F_A only through F_Aj. Consequently <f_j,K_A f_j> is a function of d_j alone, with h and J fixed; it does not depend on any other diagonal block.

This function is strictly increasing in d_j for f_j(x)=cos(x_j). Indeed, add r I_2 at coordinate j, r>0, leaving the other loadings fixed. The comparison just proved applies. Given the lower field, the additional noisy observation has a positive density on R^2. The higher posterior mean of cos(X_j) is a nonconstant function of the higher field: its derivative with respect to the cosine field is Var_VM(cos(X_j))>0 at every finite field. The conditional variance of that posterior mean given the lower field is therefore positive. The law has full support and finite fields, including when d_j=0. Thus E(m_2-m_1)^2>0.

If d_2j<d_1j, augment D_2 only at j up to d_1j. This augmented matrix remains admissible, and its coordinate-j covariance equals that for D_1 by the marginal argument. Strict increase therefore gives <cos(X_j),K_A2 cos(X_j)><cos(X_j),K_A1 cos(X_j)>. Condition 2 fails. This proves necessity.

## Implication for allocations by vertex weights

Take a connected norm graph s_jk=||M_jk||op>0 on its edges, and positive vertex weights u. The edge-admissible endpoint allocations are a_jk=s_jk u_k/u_j, giving d_j(u)=sum_k s_jk u_k/u_j. Let u and w be positive and not proportional. Write r_j=w_j/u_j. At a boundary vertex of the maximum-ratio set, d_j(w)<d_j(u); at a boundary vertex of the minimum-ratio set, d_j(w)>d_j(u). Such boundary vertices exist by connectedness. The theorem implies that K_u and K_w are incomparable in one-step covariance order. Both conclusions already follow using single-coordinate cosine functions, although sufficiency concerns every joint function f(X).

Thus minimizing the maximum loading by Perron weights does not minimize all latent prediction risks or maximize all redraw innovations. This strengthens the interpretation of the earlier minimax calculation; it is not a refutation of a theorem of power, and the older unfavorable selection experiments remain separate evidence.

For the residual Fisher-information budget in SP08, larger coordinatewise loadings decrease I_res,N/2. At the same time, they decrease E[(f(X)-f(witness))^2]=2 E Var(f(X)|F_A) for every f. A smaller upper bound obtained by revealing more latent information is therefore not a free improvement in the coupling. Neither B, Omega nor BVB^T is ordered by this argument, and no change in actual FDR or power is inferred.

## Cost, tuning and scope

The proposition compares existing kernels; it adds no operation to their O(n(p+|E|)) edge generation cost. Computing a dense bridge for a numerical check costs O(p^3) once and O(np^2) for the sampled fields; this is not the cost of the sparse production generator. The result allows anisotropic matrices, nonzero fields and singular admissible A. It excludes nonscalar quadratic self-terms and changes of the underlying h or J between the compared laws.

Loadings must be fixed, or chosen using an independent covariate-only tuning sample. Conditional on such a tuning sample, known-law exchangeability still holds if the true h,J are used. Replacing them by fitted parameters is the separate learned-law problem; no exact guarantee follows from tuning alone. An ordering of latent reconstruction or covariance cannot justify response-driven tuning on the evaluated data.

## Sources and what was read

Hobert and Marchev (2008), A theoretical comparison of the data augmentation, marginal augmentation and PX-DA algorithms, Annals of Statistics 36(2), 532-554, DOI 10.1214/009053607000000569. The electronic reprint arXiv:0804.0671 was consulted in §§2-3, in particular the covariance order and the identity connecting the operator quadratic form to conditional-expectation variance. Their general ordering framework is prior work. Their comparisons of latent sandwich moves are not asserted to be this particular Gaussian diagonal-loading characterization.

Blackwell (1953), Equivalent Comparisons of Experiments, Annals of Mathematical Statistics 24(2), 265-272, DOI 10.1214/aoms/1177729032. Bibliographic record and abstract identified; full primary text was not yet accessible through the first publisher URL. The note proves its Gaussian random-transformation comparison directly and does not claim to have audited all of Blackwell's theorem hypotheses.

The Perron minimax and endpoint-allocation results are already proved and tested in docs/v2/THEORIE_FR.md and tests/test_loading_v2.py. They are not new results of this step. The characterization above is new in this project; its bibliographic priority has not been established by a targeted search.

## Executed numerical checks

The independent script `scripts/statistical_papers/sp09_loading_order_check.py` imports no project sampler, and used no responses. The frozen run is `results/statistical_papers/sp09_loading_20261010T155304_010893a9/`, complete in 1.457 seconds. Five one-angle cases each use 100000 iid original/witness pairs; a distinct anisotropic, nonzero-field three-angle star uses 100000 exact iid originals, with 100387 accepted out of 587018 uniform rejection proposals (acceptance 0.1710). No selection dataset or MCMC sequence is generated.

All fifteen first-, second- and third-harmonic Rice integrals are normalized and increase strictly with loading. The independent pair correlations differ from the quadrature by at most 1.473 Monte Carlo standard errors. The 22-dimensional conditional-moment identity has a maximum entrywise discrepancy of 2.918 standard errors. The positive covariance-gain Gram matrix has minimum eigenvalue 0.006339. These diagnostic error ratios are not a proof or a simultaneous confidence certificate.

The baseline augmented matrix has rank five; its Gaussian bridge range defect is 9.534e-16 and its minimum conditional-covariance eigenvalue is -1.388e-17, within floating-point roundoff. Adding 0.4 to every vertex loading reduces embedded-vector innovation by 0.13887, 0.20781 and 0.20445 at the three vertices. For balanced minus symmetric loading, innovation increases by 0.13745 at the centre (paired MCSE 0.00164) and decreases by 0.12513 and 0.12194 at the leaves (0.00151, 0.00152). The raw angles and fields, fifteen quadrature rows, three star rows, matrix diagnostics, executed source and protocol hashes are retained. These numbers describe a known-law coupling diagnostic, not FDR or power.
