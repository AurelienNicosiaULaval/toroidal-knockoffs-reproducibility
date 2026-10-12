# Growing signals, consistent fields and relative coupling accuracy

Scientific revision SP13, 10 October 2026. This note supersedes the sufficient \(n/(K^3m)\) regime in the preserved SP12 note. It proves a weaker sufficient regime \(n/(K^2m)\to\infty\). It is not a finite-sample calibration, an optimal rate claim or a general positive high-dimensional theorem. The generic shared-error mechanism is inherited; bibliographic priority for the realization below is not established.

## Statement

Fix \(g>0\) and known \(\kappa=g^2\), sufficiently small for Supplementary Lemma S3 of SP11. In each of \(K=K_n\) independent pairs let
\[
 X_{2j}\sim{\rm VM}(0,\kappa),\quad U_j\sim{\rm VM}(0,g),\quad
 X_{2j-1}=X_{2j}+U_j\pmod {2\pi}.
\]
Roots and increments are independent. Subjects are iid. An independent pilot of \(m=m_n\) subjects estimates the single tied interaction \(g\) from its \(Km\) increments by its nonnegative MLE \(\widehat g\); the known field \(\kappa\) is held fixed when the working interaction changes. For selection subjects,
\[
 Y=\sum_{j=1}^K\cos(2X_{2j-1})+\xi,\qquad
 \xi\sim N(0,\sigma^2),\quad0<\sigma^2<\infty.
\]
Generate witnesses by the Gaussian conditional-redraw kernel at \(\widehat g\), and use centered first-harmonic ridge with a fixed positive penalty. Let \(q\in(0,1/4)\), \(K\to\infty\), \(m\to\infty\), and
\[
 \frac{n}{K^2m}\longrightarrow\infty.
\tag{P1}
\]
Then the fitted-law knockoff+ filter has
\[
 {\rm FDR}\to\frac14,\qquad {\rm Power}\to1,\qquad
 \Pr(\widehat S=\varnothing)\to0.
\tag{P2}
\]
The specified true-parameter filter retains exact finite-sample FDR at most \(q\), and its power tends to one. Here all odd angles are alternatives and all even angles are null, so the signal fraction is one half.

Write \(\mathfrak a=w_2'(g)<0\) for the one-pair null norm-score derivative and \(I_g=A'(g)>0\), where \(A(g)=I_1(g)/I_0(g)\). For any vector \(W^*\) whose \(K\) null coordinates have a marginal law invariant under arbitrary coordinate sign flips, and for every coupling to the fitted scores, define
\[
 D_n=\max_{j\leq K}|\,\widehat W_{2j}-W^*_{2j}\,|.
\]
For every fixed \(c\geq0\),
\[
 \liminf\Pr\{\sqrt{Km}D_n>c\}
 \geq 2\Phi\!\left(-\frac{c\sqrt{I_g}}{|\mathfrak a|}\right).
\tag{P3}
\]
Zeros in the reference vector are allowed. In particular, any deterministic accuracy \(b_n\) for which \(\Pr(D_n>b_n)\to0\) must satisfy \(b_n\sqrt{Km}\to\infty\). This is a necessary lower scale, not a sufficient coupling construction.

Absolute coupling does hold for this example. Generate true-parameter witnesses on the same selection subjects, independently of the working-kernel witnesses conditional on the originals. For the resulting oracle score vector,
\[
 \max_{j\leq2K}|\widehat W_j-W^{\rm or}_j|
 =O_{\Pr}\{K/\sqrt n+(Km)^{-1/2}\}=o_{\Pr}(1).
\tag{P3a}
\]
Thus a deterministic shrinking high-probability accuracy exists, even though no such accuracy is small on the specified oracle's fluctuation scale.

For the specified true-parameter ridge statistic there is a constant \(\alpha>0\) such that
\[
 \sqrt{\frac nK}\,W^{\rm or}_{2}\ \Rightarrow\ N(0,\alpha).
\tag{P4}
\]
Consequently every accuracy \(b_n\) as above obeys \(b_n/\sqrt{K/n}\to\infty\). With \(G_n(t)=\Pr(W^{\rm or}_{2}>t)\) and \(t_n=\sqrt{\alpha K/n}\),
\[
 \frac{G_n(t_n-b_n)-G_n(t_n+b_n)}{G_n(t_n)}
 \longrightarrow \frac1{1-\Phi(1)}>0.
\tag{P5}
\]
Thus a shrinking absolute coupling error cannot also be small relative to this oracle's null tail window. This conclusion concerns this oracle, not every possible oracle marginal distribution.

## Proof: uniform regression error

At a fixed working value \(t\) near \(g\), let \(d_t\in\mathbb R^8\) be the original/witness first-harmonic vector for one pair, \(z=\cos(2X_1)\), and put
\[
 \mu_t=\mathbb Ed_t,\quad G_t=\operatorname{Var}(d_t),\quad
 b_t=G_t^{-1}\operatorname{Cov}(d_t,z),\quad
 r_t=z-\mathbb Ez-(d_t-\mu_t)^Tb_t.
\]
The joint angular law has positive density on its full support. Hence \(G_g>0\). Continuity on a sufficiently small compact interval \(J\subset(0,\infty)\) about \(g\) gives a uniform positive lower eigenvalue, bounded \(b_t\) and bounded \(r_t\). The conditional fields are smooth on \(J\). All constants below are uniform in \(t\in J\), and do not depend on \(K,m,n\).

For \(K\) pairs, \(\Gamma_{K,t}\) is block diagonal with \(K\) copies of \(G_t\), \(b_{K,t}\) concatenates \(K\) copies of \(b_t\), and the regression residual is \(R=\sum_j r_{j,t}+\xi\). Pair independence and bounded centered residuals give
\[
 \|b_{K,t}\|\leq C\sqrt K,\quad
 \mathbb ER^2\leq CK,\quad \mathbb ER^4\leq CK^2.
\]
Put \(u_i=d_i-\mu_{K,t}\). With bars denoting empirical means, write
\[
 \widehat\Gamma=n^{-1}\sum_i u_i u_i^T-\bar u\bar u^T,\qquad
 \widehat\gamma=n^{-1}\sum_i u_iR_i-\bar u\bar R.
\]
Population orthogonality gives \(\mathbb E(uR)=0\), and bounded features give
\[
 \|\widehat\Gamma-\Gamma_{K,t}\|_{\rm op}
 =O_{\Pr}(K/\sqrt n+K/n),\quad
 \|\widehat\gamma\|=O_{\Pr}(K/\sqrt n+K/n).
\tag{P6}
\]
Indeed the covariance's squared Frobenius error is at most \(CK^2/n\), \(\mathbb E\|\bar u\|^2\leq CK/n\), \(\mathbb E\bar R^2\leq CK/n\), and \(\mathbb E\|uR\|^2\leq CK^2\). Cauchy-Schwarz and Markov's inequality give the displayed centering orders without assuming that \(u\) and \(R\) are independent.

Under (P1), \(K/\sqrt n\to0\); the inverse centered ridge covariance is uniformly bounded with probability tending to one. The exact normal equation is
\[
 \widehat b-b_{K,t}=
 (\widehat\Gamma+\lambda I/n)^{-1}
 \{\widehat\gamma-\lambda b_{K,t}/n\}.
\]
Therefore \(\|\widehat b-b_{K,t}\|=O_{\Pr}(K/\sqrt n)\). A block norm difference is Lipschitz, so
\[
 \max_j|\widehat W_j-w_j(t)|=O_{\Pr}(K/\sqrt n)=o_{\Pr}(1),
\quad
 \sum_{j\leq K}|\widehat W_{2j}-w_2(t)|^2
 \leq2\|\widehat b-b_{K,t}\|^2.
\tag{P7}
\]
The estimates are conditional on a pilot in \(J\). Since \(\widehat g\to g\), the same bounds hold unconditionally.

Apply (P7) also to the true-parameter witness draw on the same originals and responses. Smoothness of the fixed one-pair population norm scores gives \(\max_j|w_j(\widehat g)-w_j(g)|=O_{\Pr}((Km)^{-1/2})\), with a constant independent of \(K\). The triangle inequality proves (P3a), without requiring independence of the two regression errors. For \(r_n=K/\sqrt n+(Km)^{-1/2}\to0\), one may take accuracy \(\sqrt{r_n}\): its ratio to \(r_n\) diverges and its failure probability tends to zero.

## Proof: most null signs suffice

The pilot uses \(Km\) independent increments with information \(I_g\), so
\[
 \sqrt{Km}(\widehat g-g)\Rightarrow N(0,I_g^{-1}).
\]
Truncation at zero is inactive with probability tending to one. Lemma S3 gives \(w_2(g)=0\), \(\mathfrak a\ne0\), and \(w_1(g)>0\). Set \(C_n=\sqrt{Km}\,w_2(\widehat g)\). Then \(C_n\Rightarrow C=\mathfrak a Z/\sqrt{I_g}\), \(Z\sim N(0,1)\), and (P7) implies
\[
 M_n:=\frac1K\sum_{j\leq K}
 \{\sqrt{Km}\widehat W_{2j}-C_n\}^2
 =O_{\Pr}(K^2m/n)=o_{\Pr}(1).
\tag{P8}
\]
For fixed \(\epsilon,\eta>0\), on \(C_n>\epsilon\), the fraction of nonpositive fitted null scores is at most \(M_n/\epsilon^2\). On \(C_n<-\epsilon\), the fraction of nonnegative null scores has the same bound. Thus almost every null sign agrees with \(C_n\). An assertion that all \(K\) signs agree is unnecessary and is not claimed under (P1).

By (P7), all active scores are bounded away from zero and exceed all absolute null scores with probability tending to one. A threshold at the smallest active score has ratio \(1/K\leq q\); hence all \(K\) alternatives are selected. On \(C_n>\epsilon\), there are \(K-o_{\Pr}(K)\) positive nulls and \(o_{\Pr}(K)\) negative nulls. The smallest nonzero absolute score has threshold ratio tending to zero and is admissible, so every positive score is selected. FDP tends to \(1/2\) on this event. On \(C_n<-\epsilon\), at most \(o_{\Pr}(K)\) positive nulls can be selected and FDP tends to zero. The probability \(\Pr(|C_n|\leq\epsilon)\) approaches the corresponding Gaussian probability. Letting \(\epsilon\downarrow0\), boundedness of FDP gives (P2). The same active/null separation applies at \(t=g\); exact known-law exchangeability gives its finite-sample FDR bound.

## Proof: all-coupling lower bound

For an exact sign-flip null vector, conditional on its absolute values, signs of nonzero coordinates are independent fair signs. With zero coordinates allowed, the positive count and the negative count are each stochastically bounded by \({\rm Bin}(K,1/2)\). For any fixed \(0<\eta<1/2\), Hoeffding's bound therefore yields
\[
 \Pr\{\max(N_+(W^*),N_-(W^*))\geq(1-\eta)K\}
 \leq2\exp\{-2K(1/2-\eta)^2\}.
\tag{P9}
\]
This bound holds for the reference's marginal law, regardless of its coupling to \(C_n\).

Fix \(c\geq0\) and \(\delta>0\). On \(|C_n|>c+\delta\), at least \(K(1-M_n/\delta^2)\) fitted null scores have its sign and scaled magnitude exceeding \(c\). If \(\sqrt{Km}D_n\leq c\), the reference must have that same sign at all these coordinates. Except on \(M_n/\delta^2>\eta\), this entails the event in (P9). Subtract its probability and \(\Pr(M_n>\eta\delta^2)\), then pass to the limit and let \(\delta\downarrow0\). This proves (P3), including \(c=0\). If a deterministic \(b_n\sqrt{Km}\) stayed bounded along a subsequence, (P3) at a larger fixed \(c\) would contradict \(\Pr(D_n>b_n)\to0\).

## Proof: oracle fluctuation and the relative window

For the univariate oracle CLT a sharper Gram bound is needed. For a unit vector \(v\), \(v^Tu\) is a sum of \(K\) independent centered pair contributions bounded by \(C\|v_j\|\). Hoeffding's lemma gives a uniform sub-Gaussian bound because \(\sum_j\|v_j\|^2=1\). Its square is uniformly sub-exponential. Scalar Bernstein's inequality on a \(1/4\)-net of the sphere in \(\mathbb R^{8K}\), of size at most \(9^{8K}\), gives
\[
 \|\widehat\Gamma-\Gamma_{K,g}\|_{\rm op}
 =O_{\Pr}(\sqrt{K/n}+K/n).
\tag{P10}
\]
For completeness, for every net point the failure probability at deviation \(x\) is at most \(2e^{-cn\min(x^2,x)}\); take \(x=C_0\sqrt{K/n}\), use the net inequality for a symmetric matrix, and increase \(C_0\). Empirical centering contributes \(O_{\Pr}(K/n)\). Since \(K\to\infty\), this proves the asserted stochastic order.

The null common population block is nonzero for the small \(g,\kappa\) supplied by Lemma S3, so the norm score is locally differentiable. In the pair ordering original-1, original-2, witness-1, witness-2, its gradient is \(l=(0,0,-1,0,0,0,1,0)^T\). Let
\[
 \chi=l^TG_g^{-1}(d_g-\mu_g),\quad r=r_g,\quad
 \alpha=(\mathbb E\chi^2)(\mathbb Er^2)>0.
\]
Here \(\mathbb E\chi=0\) and \(\mathbb E(\chi r)=0\) by regression orthogonality. The residual variance is positive: the second harmonic cannot be a linear combination of the first-harmonic original/witness features on their full angular support. For pair one's influence \(L_1=\chi_1(\sum_j r_j+\xi)\),
\[
 \operatorname{Var}(L_1)=\mathbb E(\chi^2r^2)
 +\{(K-1)\mathbb Er^2+\sigma^2\}\mathbb E\chi^2,\quad
 \mathbb EL_1^4\leq CK^2.
\]
Rows are independent. Lyapunov's condition yields the CLT for the empirical mean influence normalized by \(\sqrt{n/K}\). By (P10) and (P6), replacing the population Gram inverse by its empirical version costs \(O_{\Pr}(K^{3/2}/n)\) in total coefficient norm. Its normalized contribution is \(O_{\Pr}(K/\sqrt n)=o_{\Pr}(1)\). Centering costs \(O_{\Pr}(K/n)\), and fixed ridge costs \(O(\sqrt K/n)\). A fixed pair's leading coefficient error is \(O_{\Pr}(\sqrt{K/n})\); the inverse remainder is smaller because \(K/\sqrt n\to0\). The quadratic norm remainder is \(O_{\Pr}(K/n)\). All vanish after the same normalization. This proves (P4).

From (P1) and (P3),
\[
 \frac{b_n}{\sqrt{K/n}}=
 \{b_n\sqrt{Km}\}\sqrt{\frac{n}{K^2m}}\to\infty.
\]
The CLT implies tightness of the normalized oracle. Its two tails at \(t_n-b_n\) and \(t_n+b_n\) tend respectively to one and zero, while \(G_n(t_n)\to1-\Phi(1)\). This proves (P5).

## Consistency and comparison to ARK

Every conditional field has uniform error \(|\widehat g-g|=O_{\Pr}((Km)^{-1/2})\). The Frobenius error of the \(K\) upper-edge \(2\times2\) matrices is \(\sqrt{2K}|\widehat g-g|=O_{\Pr}(m^{-1/2})\). The per-subject covariate KL divergence is exactly
\[
 D_K=K\{(g-\widehat g)A(g)+\log I_0(\widehat g)-\log I_0(g)\};
 \quad mD_K\Rightarrow\chi_1^2/2.
\]
These are consistency statements for a subject's law and conditional fields. The full \(n\)-subject experiment is not claimed close.

The reference comparison is to [ARK, version of 4 June 2024](https://arxiv.org/pdf/2307.04400), Conditions 1 and 5, equation (7). That condition requires relative tail-window accuracy, with an additional logarithmic factor. The failure (P5) already occurs without that factor. At \(p=2K\), all \(K\) oracle alternatives have a fixed positive population score. For its strong-signal count \(a_n=K\), any \(c_1\in(0,1)\) and \(q<1/4\), the oracle tail \(G_n(t_n)\to1-\Phi(1)>c_1q/2\), so \(t_n\) lies inside the specified positive-threshold range eventually. This verifies the location of the obstruction, not merely an arbitrary remote tail. ARK's other hypotheses are not asserted here, and the null fraction is one half rather than tending to one.

The sufficient rate (P1) is not shown necessary. No general-matrix pilot, learned graph, sparse-signal theorem, dimension comparable to sample size, response-adaptive basis or positive high-dimensional guarantee is derived. All future numerical checks must distinguish retained-data diagnostics, Gaussian limits and genuinely new iid toroidal experiments.
