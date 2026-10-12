#!/usr/bin/env python3
"""One-time, hash-guarded integration into the existing manuscript.

Numerical tables come only from the independently recomputed SP06 CSV.
The previous source is immutable and this program refuses later user edits.
It does not run simulations, compile PDFs or communicate with a journal.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / 'provenance/statistical_papers/sp08_20261010T144857_b5861c07/before'
AUDIT = ROOT / 'results/statistical_papers/hc_audit_20261010T143014_33d7d9dd'

INTRO = r"""\section{The fitted-law question}
Model-X knockoffs control the false discovery rate when the covariate law is known~\citep{candes}. With a law fitted on an independent pilot, consistency of the fit does not by itself determine calibration of a response statistic. We study whole-angle ridge blocks in a bilinear torus graph. This setting permits exact known-law witnesses, a regular compatible pilot estimator, and an explicit calculation of the finite knockoff+ threshold.

Three results describe the role of the response projection and the latent coupling. A represented additive mean has the same population regression coefficient under every working covariate law, so pilot consistency suffices at fixed dimension. An omitted harmonic permits a common estimation error to align ten null scores: a correctly specified, regular pilot then gives limiting false discovery rate $1/4$ at nominal level $0.1$. For conditional independent redraws, a score-information calculation bounds the aggregate null sensitivity by one half of the information not explained by the latent variables and the remaining original angles. The bound applies beyond independent pairs; its constant is attained in a finite-angle example. It does not imply a universal power improvement or restore exchangeability under a fitted law.

The construction combines existing ideas. Conditional independence yields valid knockoffs~\citep[Theorem 12]{berti}, with a systematic treatment by \citet{dreassi}. Gaussian circular augmentation is explicit in \citet{cohen}. Metro supplies a general graphical-model sampler~\citep{metro}, while \citet{spector} analyze reconstructability as a coupling criterion. Edge and global factorizations of the same augmented covariance give the same witness law. Scalable and dynamic torus graph fitting~\citep{goffinet} addresses a different estimation problem; our results keep the dimension and graph fixed.

Our analysis specializes statistic-specific robustness~\citep[Section 3.2.1]{bcs}. Conditional knockoffs~\citep{huang} obtain finite-sample guarantees by conditioning on sufficient statistics; the fitted-law counterexample does not concern that construction. ARK~\citep{ark} and the moment-based analysis of \citet{moments} use coupling and strong-signal assumptions. Their least-squares results require a diverging strong-signal count. Example 3 of the latter already exhibits inflation for non-Gaussian covariates with a Gaussian generator and a marginal statistic. Our calculation instead uses a correctly specified toroidal family, a regular independent estimator, block ridge scores and a fixed signal count. Technical comparisons with that paper refer to the author manuscript dated 12 June 2026; the published DOI is also supplied.

We also give studentized block contrasts that retain the independent pilot's first-order variance. Variance addition is classical two-step inference~\citep{murphy,newey}; the contribution here is the swap-based sensitivity formula and its implementation. Learned-law studentization and response approximation are already central in conditional independence testing~\citep{niu}. Benjamini--Yekutieli (BY) adjustment~\citep{by} supplies the multiple-testing step. This procedure has an asymptotic guarantee for arbitrary response means under explicit regularity, but is not the knockoff filter. Its numerical evaluation retains a known-law small-sample calibration failure and a substantial power cost. Table~\ref{tab:guarantees} separates the guarantees.

\begin{table}[htbp]\centering\small
\caption{Guarantees and their scope. An angle is one hypothesis throughout. Fitted-law conclusions are pointwise and asymptotic, with fixed dimension.}\label{tab:guarantees}
\begin{tabular}{p{.23\linewidth}p{.34\linewidth}p{.34\linewidth}}\toprule
Procedure or calculation & Conditions & Conclusion and limitation\\\midrule
Known-law knockoff+ & Correct bilinear covariate law; swap-equivariant score & Finite-sample FDR $\leq q$ for arbitrary response laws.\\
Fitted-law knockoff+ & Represented additive mean; independent homoskedastic noise; consistent pilot & Asymptotic FDR $\leq q$, with no pilot-to-selection rate restriction.\\
Shared-error example & Ten independent pairs; omitted harmonic; regular common pilot & FDR tends to $1/4$ when $m/n\to0$. No claim of widespread inflation at comparable sizes.\\
Studentized block tests & Correct regular pilot; smooth kernel; $m/n\to\rho>0$; nonsingular contrast covariance & Asymptotic BY control; projection tests are not omnibus and may lose power.\\
Latent information bound & Oracle conditional independent redraw; regular null sensitivity & Aggregate sensitivity $\preceq$ half the residual information. A covariance bound is not an FDR certificate.\\\bottomrule
\end{tabular}
\end{table}
"""

WALD = r"""\section{Retaining independent-pilot uncertainty}\label{sec:wald}
This section replaces the norm-difference filter by block tests. Let $\theta$ be the vector of fields and compatible edge matrices in \eqref{eq:family}, with bounded canonical sufficient statistic $\mathcal T(X)\in\R^k$. Write $\beta(\theta,\eta)$ for the centered population regression coefficient under $P_\theta K_\eta$, with the conditional response law held fixed. Let $C_j$ subtract the witness block from the original block, and define at the oracle diagonal
\begin{align}\label{eq:influences}
 \Gamma&=\Var(d),\quad R=Y-\E Y-(d-\E d)^T\beta(\theta_0,\theta_0),\notag\\
 L_j&=C_j\Gamma^{-1}(d-\E d)R,\quad
 \Omega_j=\E L_jL_j^T,\quad
 B_j=-\E[L_j\{\mathcal T(X)-\E\mathcal T(X)\}^T].
\end{align}
Under $H_{0j}$, a null swap preserves the joint law including $Y$, so uniqueness of the population regression equates the two coefficient blocks. Thus $C_j\beta(\theta,\theta)=0$ for every nearby $\theta$, even if the response mean is not represented. Differentiating this identity along the diagonal gives
\begin{equation}\label{eq:sensitivity}
 \left.\partial_\eta\{C_j\beta(\theta_0,\eta)\}\right|_{\eta=\theta_0}=B_j.
\end{equation}
Indeed, differentiating the centered normal equations with respect to the original distribution gives $\E[L_j(\mathcal T-\E\mathcal T)^T]$. Mean-derivative terms vanish by the regression orthogonality equations. The diagonal derivative is zero, proving \eqref{eq:sensitivity}. Estimating $B_j$ requires a cross-covariance, not the global normalizer.

\begin{theorem}[Pilot-variance block tests]\label{thm:wald}
Fix $p,H$, a containing graph, and an identifiable canonical covariate family with interior true parameter $\theta_0$. Assume $\E|Y|^{2+\delta}<\infty$ for some $\delta>0$. The valid kernels $K_\eta$ and population normal equations have locally continuous first two derivatives; the centered design covariance is positive definite locally, and the conditional regression central limit theorem and covariance laws of large numbers hold locally in $\eta$. Let the independent covariate-only pilot satisfy
\[
 \sqrt m(\widehat\theta-\theta_0)\Rightarrow N(0,V),\qquad
 \widehat V\to V,\qquad m/n\to\rho\in(0,\infty].
\]
Retain this root-$m$ regularity when $\rho=\infty$, and use $\lambda_n=o(\sqrt n)$. For a null angle let $S_j=\Omega_j+\rho^{-1}B_jVB_j^T$ be positive definite, with $\rho^{-1}=0$ at infinity. Then, for $\widehat q_j=C_j\widehat b$,
\[
 \sqrt n\widehat q_j\Rightarrow N(0,S_j),\qquad
 \widehat q_j^T\widehat\Sigma_j^{-1}\widehat q_j\Rightarrow\chi^2_{2H},\qquad
 \widehat\Sigma_j=\widehat\Omega_j/n+\widehat B_j\widehat V\widehat B_j^T/m.
\]
Here empirical residual influences and their cross-covariances consistently estimate \eqref{eq:influences}. Apply BY thresholds $\alpha_r=rq/(p\sum_{l=1}^p1/l)$ to the resulting chi-square tail probabilities. The procedure satisfies $\limsup\operatorname{FDR}\leq q|H_0|/p\leq q$.
\end{theorem}
\begin{proof}
Conditional on the pilot, regression fluctuations centered at $\beta(\theta_0,\widehat\theta)$ converge to an oracle Gaussian law with nonrandom covariance. Conditional characteristic functions give joint convergence and limiting independence from the pilot error. Taylor expansion of a null population contrast gives $B_j(\widehat\theta-\theta_0)$; its root-$n$ remainder is $O_{\Prob}(\sqrt n/m)=o_{\Prob}(1)$. The ridge contribution is $O_{\Prob}(\lambda_n/\sqrt n)=o_{\Prob}(1)$. Adding the independent limits proves the covariance formula. Conditional laws of large numbers give $n\widehat\Sigma_j\to S_j$, so Slutsky's theorem gives the chi-square limit. For each null, its rejection contribution is bounded by $\sum_{r=1}^p\{F_j(\alpha_r)-F_j(\alpha_{r-1})\}/r$, where $\alpha_0=0$ and $F_j$ is its null p-value distribution function. At these finitely many thresholds $F_j$ converges to the uniform distribution function. Summing yields $q|H_0|/p$. This is the usual BY argument.
\end{proof}

The estimated sensitivity uses evaluated responses as part of studentization; it does not tune the covariate fit or choose a basis. A correctly represented conditional mean makes $B_j=0$, since the regression residual has conditional mean zero given originals and witnesses. A dependent alternative can nevertheless have $q_j=0$ in the chosen projection, so the tests are not omnibus. Correct covariate specification, smoothness, regularity and nonsingularity are substantive assumptions. A learned graph, increasing dimension, $\rho=0$ and singular variances are outside this theorem.

Compatible pilot fitting minimizes the joint negative pseudolikelihood
\begin{equation}\label{eq:pl}
 \frac1m\sum_{i,j}\{\log[2\pi I_0(\|\eta_j^{(i)}\|)]
 -(\eta_j^{(i)})^Tv(X_j^{(i)})\}
 +\frac{\tau_m}{2}\|\theta\|^2,
 \quad \eta_j=h_j+\sum_kM_{jk}v(X_k),
\end{equation}
with one matrix per undirected edge and $M_{kj}=M_{jk}^T$. At fixed graph, $\tau_m\to0$ gives consistency; $\sqrt m\tau_m\to0$ gives the regular covariance $V=\mathcal H^{-1}\mathcal K\mathcal H^{-1}$, where $\mathcal H$ is the population loss Hessian and $\mathcal K$ is the covariance of the \emph{full subject} loss gradient. Cross-node score covariances must be retained. Supplementary Theorem S9 proves this statement and treats numerical residuals. The new star experiments use $\tau_m=1/m$; the earlier implementation's $m^{-2}$ penalty remains separately documented.

Smoothness holds locally for the implemented edge loading when every nonzero edge has a simple largest singular value separated from the second one. Its constant-rank augmented matrix admits a smooth square root; numerical eigenvector signs do not matter because the field law depends only on $A$. Equal largest singular values, zero edges and graph changes require a different argument. Prespecified strict scalar loadings on a compact parameter neighborhood give another smooth kernel, but are not evaluated here. The supplement supplies the local proof.
"""

INFORMATION = r"""\section{An information budget for latent redraws}\label{sec:latent}
Under the oracle law, a null swap changes $L_j$ to $-L_j$ and leaves every other null influence unchanged. Thus their covariance is block diagonal. For any null subset $N$, let $L_N$ and $B_N$ stack the corresponding blocks and put $\Omega_N=\operatorname{diag}(\Omega_j:j\in N)$. Write $\mathcal I=\Var\{\mathcal T(X)\}>0$. The ordinary covariance projection inequality gives $B_N^T\Omega_N^{-1}B_N\preceq\mathcal I$. Conditional independent redraws obey a stronger bound.

\begin{theorem}[Half-information bound]\label{thm:half}
Suppose \eqref{eq:sensitivity} holds and $\Omega_N>0$. Suppose a latent variable $Z$ makes all original coordinates conditionally independent, and witnesses are independent draws from those same conditional coordinate laws. Define
\[
 \mathcal I_N^{\rm res}=\E\Var\{\mathcal T(X)\mid Z,X_{-N}\}.
\]
Then
\begin{equation}\label{eq:half}
 B_N^T\Omega_N^{-1}B_N\preceq\tfrac12\mathcal I_N^{\rm res}
 \preceq\tfrac12\mathcal I.
\end{equation}
For any pilot covariance $V\succeq0$, set
$\kappa_V=\lambda_{\max}(\mathcal I^{1/2}V\mathcal I^{1/2})$. Then
\[
 B_NVB_N^T\preceq\tfrac12\kappa_V\Omega_N,\qquad
 \Omega_N+\rho^{-1}B_NVB_N^T
 \preceq\left(1+\frac{\kappa_V}{2\rho}\right)\Omega_N.
\]
\end{theorem}
\begin{proof}
Let $X^N$ replace precisely the original coordinates in $N$ by their witnesses. Simultaneously swapping the null subset sends $L_N$ to $-L_N$, preserves the oracle triple law, and sends $\mathcal T(X)$ to $\mathcal T(X^N)$. Therefore, for $\Delta=\mathcal T(X)-\mathcal T(X^N)$,
$\E L_N\Delta^T=-2B_N$. Conditional on $(Z,X_{-N})$, the two sufficient-statistic vectors are iid. Hence $\Var(\Delta)=2\mathcal I_N^{\rm res}$. The Schur complement of the covariance matrix of $(L_N,\Delta)$ yields
$4B_N^T\Omega_N^{-1}B_N\preceq2\mathcal I_N^{\rm res}$.
Total variance gives $\mathcal I_N^{\rm res}\preceq\mathcal I$.
Whitening implies $B_N\mathcal I^{-1}B_N^T\preceq\Omega_N/2$; using $V\preceq\kappa_V\mathcal I^{-1}$ proves the last displays.
\end{proof}

The proof applies to the Gaussian augmentation in \eqref{eq:kernel}, including dependent general matrices. It also applies to scalar smooth score influences, such as the null norm differences in Theorem~\ref{thm:shared}. It uses established score differentiation and covariance projection identities. No bibliographic priority for a general information inequality is asserted. For an efficient likelihood pilot, $V=\mathcal I^{-1}$ and $\kappa_V=1$. A regular joint pseudolikelihood generally has a larger covariance; the factor must not silently be set to one.

The constant $1/2$ is attained in a bounded finite-angle canonical family. Take two angles on $\{0,\pi/2,\pi,3\pi/2\}$, $\mathcal T(x)=\cos x_1\cos(2x_2)$, $P_\theta\propto\exp\{\theta\mathcal T\}$ and $Y=\cos(2X_2)$ at $\theta=0$. Coordinate 1 is null. Conditional on $Z=\cos(2X_2)$, independently redraw both coordinates from their conditional laws. For $H=1$, $\Gamma=I_8/2$, $\mathcal I=1/2$, $\Omega_1=4I_2$ and $B_1=(-1,0)^T$, giving equality. A valid negatively correlated pair, with conditional cosine copy correlation $-1/4$, instead gives $B_1^T\Omega_1^{-1}B_1/\mathcal I=5/8$. It satisfies the general bound but is outside the redraw class. The supplement proves validity and independently enumerates all 256 joint states. Sharpness is not claimed within the continuous bilinear torus subfamily.

\begin{corollary}[Restricted sufficient pilot ratio]\label{cor:ratio}
Under every premise of Theorem~\ref{thm:shared}, put $c=a^2V/\tau^2$. Then $c\leq1/20$. The limiting FDR satisfies
\[
 F_\rho\leq F_0+\frac1{16\rho},\qquad
 F_0=\frac{467686735}{5959489536}\simeq0.07847765.
\]
Consequently $\rho\geq3$ suffices for asymptotic FDR below $0.1$ in that counterexample.
\end{corollary}
\begin{proof}
There are ten orthogonal scalar null influences and an efficient scalar common pilot. Theorem~\ref{thm:half} gives $10a^2V/\tau^2\leq1/2$. The limiting Gaussian covariance has a single inflated eigenvalue $1+10c/\rho$. Differentiation of a centered normal density with respect to log variance gives total variation at most $\varphi(1)\log(1+t)\leq t/4$ for variance inflation $1+t$, where $\varphi$ is the standard normal density. The leading-run FDP lies in $[0,1/2]$, so its expectation changes by at most half this total variation. This gives $F_\rho\leq F_0+1/(16\rho)$. The exact rational value $F_0+1/48$ is below $1/10$.
\end{proof}
This refines the general-information ratio six retained in the earlier calculation. It is sufficient, not necessary, and concerns only the stated asymptotic counterexample. Greater latent predictability reduces $\mathcal I_N^{\rm res}$, but may also reduce the response contrast; it is not a power ordering. Neither \eqref{eq:half} nor an estimated $B_j$ alone certifies finite-sample FDR.
"""

NUMERICAL_START = r"""\section{Numerical evaluation}\label{sec:numerical}
The older independent-pair studies and the newer general-matrix star studies have separate configurations, code snapshots and output directories. We report FDR as the mean of individual FDPs, power as the unconditional mean fraction of signals selected, and the probability of an empty selection. Monte-Carlo standard errors (MCSEs) use independent repetitions; method differences use paired repetitions. Every configuration below has 1000 repetitions. No experimental response is used to fit or tune the covariate model.

\subsection{The independent-pair mechanism}
The retained seven configurations in Studies A and B contain 7000 datasets, reaggregated from 36000 saved selection decisions. At $g=\kappa=1$, $n=320000$, $m=20$, pooled first-harmonic fitting gives FDR $0.1330$ (MCSE $0.0051$); $H=2$ gives $0.0731$ ($0.0030$). These parameters are outside the analytic small-interaction premises. With $m=n=80000$, pooled FDR is $0.0847$ ($0.0033$). The retained influence calibration $c\simeq2.70\times10^{-5}$ yields a weak common component at $m=n$; this is not evidence of inflation at comparable sample sizes.

Representing the omitted harmonic removes the population drift. Harmonic enlargement itself is standard~\citep{dai}, and can cost power when the smaller basis suffices. With a represented first-harmonic mean and $n=m=400$, known-law power falls from $0.5624$ to $0.2804$, a paired change of $-0.2820$ (MCSE $0.0177$). In the retained second-harmonic-response configurations, $H=1$ power is at most $0.004$, compared with approximately $0.32$--$0.54$ for $H=2$. All configurations, including these reversed comparisons, remain in the supplement and saved outputs.

\subsection{General fields, anisotropy and pilot variance}
The new simulations use a known star graph with nonzero fields and general edge matrices. For zero-based coordinates $j=0,\ldots,p-1$, let $h_0=3v(0.7)$, $h_j=0.2v(0.7+2\pi j/(p-1))$, and
$M_{0j}=R(0.17j)\operatorname{diag}(0.8,0.5)R(-0.11j)^T$ for $j>0$, where $R(t)$ is planar rotation. Originals and pilots are iid exact draws: propose the root from its field von Mises law, reject with probability determined by $\prod_{j>0}I_0(\|h_j+M_{0j}^Tv(X_0)\|)$ and its uniform envelope, then independently draw the conditional leaves. The envelope uses $\|h_j\|+\|M_{0j}\|_{\rm op}$. Recorded acceptance is about $0.29$; no MCMC states are used as subjects.

All fields and edge coordinates are fitted jointly, with 68 parameters at $p=12$ and 116 at $p=20$. The penalty in \eqref{eq:pl} is $1/m$, and response ridge uses $\lambda=1$. Pilot Hessians and full subject score covariances give $\widehat V$. The initial Wald estimate uses empirical residual influences (HC0). A separate leverage sensitivity replaces its residuals by $\widehat R_i/(1-h_i)$, where $h_i$ includes the intercept and fixed ridge leverage, retaining the unpenalized influence precision. It leaves $\widehat B_j$ unchanged. At zero ridge this is the usual OLS HC3 formula~\citep{hc3}; at positive ridge we call it \emph{HC3-style}, not an exact ridge jackknife. Fixed dimension makes the maximum leverage vanish asymptotically, so it has the same limit. No finite-sample guarantee is asserted for either variant.

Five initial $p=12$ configurations generated 5000 datasets; a large-$n$ follow-up generated 1000. Their six signals cannot alone reach the ten-discovery floor of knockoff+ at $q=0.1$. In the strong-signal limit for this configuration, oracle power is $1/16$ and the empty-selection probability is $15/16$, an ordinary threshold effect proved in Supplementary Proposition S11. We therefore do not treat their Wald power advantage as evidence of a superior testing method.

A prespecified sensitivity reanalysis used the same 5000 datasets with $H=1,2$ and HC0 or HC3-style, without adding independent repetitions. Table~\ref{tab:null} retains the global-null configuration. Even with the true covariate law, the $H=2$ HC0 Wald procedure has FDR $0.193$; adding pilot variance cannot explain or solve this known-law discrepancy. The leverage variant is conservative. This finite-sample failure limits practical interpretation of Theorem~\ref{thm:wald}.
"""

NUMERICAL_END = r"""A subsequent protocol fixed $p=20$, ten signals, $H=2$ and HC3-style as the principal comparison before these outcomes were generated. The choice followed the six-signal discovery-floor finding, rather than being part of the initial plan. Three configurations generated 3000 new datasets. Their matched harmonic/leverage reanalysis adds no datasets. For $s=10$ active coordinates, the continuous response has mean $3s^{-1/2}\sum_{j<s}\cos\{2(X_j-0.7)\}$ plus standard normal noise. The binary response has logit $-0.25+1.5s^{-1/2}\sum_{j<s}\cos\{2(X_j-0.7)\}$. Its probability is not an additive harmonic mean. All methods use nominal $q=0.1$.

Table~\ref{tab:fair} reports the principal comparison. The continuous mean is represented at $H=2$, so its true $B_j$ vanishes; near-unit power does not establish a benefit from pilot-variance correction. For the binary response, corrected Wald power is $0.4248$, versus $0.6066$ without the pilot component and $0.4738$ for fitted-law knockoff+. The paired power changes are $-0.1818$ (MCSE $0.0049$) and $-0.0490$ ($0.0164$), respectively. All five FDR estimates are below $0.1$. The correction trades power for substantial conservatism; these configurations demonstrate no practical FDR repair because the uncorrected comparator is also conservative. The complete 16-method sensitivities and all individual records are retained.

Across the new star studies there are 9000 distinct datasets: 5000 initial, 1000 large-$n$, and 3000 ten-signal follow-up. The two matched reanalyses total 8000 reused datasets and are not added to this count. Recorded main-run elapsed times are 198.1, 340.2 and 439.8 seconds, respectively; matched reanalyses take 141.3 and 268.7 seconds. These are measured on the recorded machine and exclude later audits and packaging. No numerical failure occurred in these runs. Fifty-one designated seed replays reproduced their respective stored outputs exactly. A separate program recomputed all 128000 decisions in the two matched reanalyses and recovered the aggregate and paired summaries. The older 7000 pair-study datasets were reaggregated, not all resimulated.
"""

DISCUSSION = r"""\section{Discussion and reproducibility}
The results identify a fixed-dimensional boundary. With a represented additive mean and independent homoskedastic noise, the fitted-law ridge target does not drift, and consistency is enough for the finite knockoff+ rule asymptotically. With an omitted mean component, a regular common pilot can create correlated null signs. The independent-pair example proves failure of consistency alone, while its small calibrated sensitivity shows why that failure need not be practically large at comparable sample sizes.

The conditional-redraw bound extends the sensitivity analysis to dependent graphs. It limits the aggregate influence projection through residual information after latent conditioning. It does not rank valid couplings by power. Similarly, retaining pilot uncertainty in a Wald contrast gives a pointwise asymptotic test, but numerical conservatism and the known-law HC0 failure prevent a general recommendation. BY adjustment, two-step variance addition and harmonic enrichment are inherited tools. The narrower contributions are the regular fitted-law finite-filter calculation, the null-swap sensitivity specialization and the conditional-redraw information budget.

High-dimensional graphs, data-driven support selection for the block-test theorem, misspecified covariate conditionals and a finite-sample fitted-law certificate remain unresolved. A genuine dependent alternative may also be invisible to a fixed projection. The earlier handwritten-digit illustration is retained as descriptive archival material; discretization and unidentified repeated writers preclude a population FDR certification. It is not used as empirical validation here. An independent-unit scientific application remains a limitation rather than a requirement claimed to have been met.

The supplementary document supplies the complete original proofs and the new regularity, variance and information calculations. Python is the executed reference implementation. Each experimental version records its configuration, individual outputs, code hashes, source snapshots and separate timings. Numerical tables here are constructed from an independent audit of saved results. The finite-angle sharpness check enumerates 256 states and verifies the population derivative by a separate finite difference; it is not a continuous torus or FDR simulation. The reproduction archive separates old studies, newly executed star studies, matched reanalyses and checks. Rebuilding summaries does not rerun their simulations.

\section*{Statements and Declarations}
\paragraph{Funding.} This research received no external funding.
\paragraph{Competing interests.} The author declares no competing interests.
\paragraph{Data availability.} This article uses simulated data. Configurations, code, recorded individual outputs and their aggregate checks are supplied with the accompanying reproducibility archive. No new participant-level dataset is analyzed in this article.
\paragraph{Computational assistance.} OpenAI Codex and Anthropic Claude assisted with mathematical arguments, code and manuscript preparation. The author takes responsibility for the final content. The earlier material was reviewed and edited by the author; the revisions reported here remain a draft for the author's final review.
"""

ABSTRACT = r"""An independent covariate pilot can be consistent without preserving false discovery rate control for a fitted-law knockoff filter. We examine whole-angle ridge scores for dependent angular covariates in a bilinear torus graph. At fixed dimension, representing an additive response mean makes the population regression target independent of the working covariate law; pilot consistency then suffices for asymptotic control without a relative sample-size condition. With an omitted harmonic, a regular common maximum likelihood estimate aligns ten null scores and gives limiting false discovery rate one quarter at nominal level one tenth. For conditionally independent latent redraws, we bound the aggregate null sensitivity by one half of the residual Fisher information after latent conditioning. The constant is attained in a finite-angle family, and the bound yields a restricted sufficient pilot ratio in the counterexample. Studentized block contrasts retain independent-pilot variance through a swap-based derivative identity and classical two-step inference. Simulations with nonzero fields, anisotropic matrices and binary responses retain a known-law small-sample Wald calibration failure and a substantial power cost of variance correction. These results provide specified asymptotic guarantees and failure mechanisms, rather than a finite-sample certificate for learned covariate laws or a universal improvement in power."""

NEW_BIB = r"""
\bibitem[Benjamini and Yekutieli(2001)]{by} \textsc{Benjamini, Y. and Yekutieli, D.} (2001).
The control of the false discovery rate in multiple testing under dependency.
\emph{Ann. Statist.}, 29, 1165--1188.
\url{https://doi.org/10.1214/aos/1013699998}.

\bibitem[MacKinnon and White(1985)]{hc3} \textsc{MacKinnon, J. G. and White, H.} (1985).
Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties.
\emph{J. Econometrics}, 29, 305--325.
\url{https://doi.org/10.1016/0304-4076(85)90158-7}.

\bibitem[Murphy and Topel(1985)]{murphy} \textsc{Murphy, K. M. and Topel, R. H.} (1985).
Estimation and inference in two-step econometric models.
\emph{J. Bus. Econ. Statist.}, 3, 370--379.
\url{https://doi.org/10.1080/07350015.1985.10509471}.

\bibitem[Newey and McFadden(1994)]{newey} \textsc{Newey, W. K. and McFadden, D.} (1994).
Large sample estimation and hypothesis testing. In: Engle, R. F. and McFadden, D. (eds.),
\emph{Handbook of Econometrics}, vol. 4, chapter 36, pp. 2111--2245. Elsevier.
\url{https://doi.org/10.1016/S1573-4412(05)80005-4}.

\bibitem[Niu et al.(2024)]{niu} \textsc{Niu, Z., Chakraborty, A., Dukes, O. and Katsevich, E.} (2024).
Reconciling model-X and doubly robust approaches to conditional independence testing.
\emph{Ann. Statist.}, 52, 895--921.
\url{https://doi.org/10.1214/24-AOS2372}.
"""

SUPPLEMENT = r"""\section{Independent-pilot block-test details}\label{app:wald_new}
The main article's Theorem 3 is a different procedure from knockoff+. The response is regressed on the fixed original and witness harmonic blocks, with centered design $d$ and oracle residual $R$. For a null angle $j$, swap symmetry forces $C_j\beta(\theta,\theta)=0$. The conditional response law is fixed throughout differentiation and continues to depend only on $X_{-j}$ under nearby true parameters. Differentiating the centered normal equations in the true canonical parameter cancels the mean-derivative terms by $\E R=0$ and $\E[(d-\E d)R]=0$. It gives
\[
 \partial_\theta\{C_j\beta(\theta,\eta)\}|_{\theta=\eta=\theta_0}
 =\E[L_j\{\mathcal T(X)-\E\mathcal T(X)\}^T]=-B_j.
\]
The derivative in the kernel parameter is the opposite one. This identity uses the known-law diagonal only; a fitted-law sample need not be exchangeable.

For full precision, the assumptions are fixed $p,H,k$ and a fixed containing graph, an interior identifiable canonical parameter with bounded $\mathcal T$, $\E|Y|^{2+\delta}<\infty$, positive definite local design covariance, locally continuous twice differentiated normal equations, and locally valid conditional covariance laws of large numbers and regression central limit theorems. Smooth parameterized kernels with dominated moment derivatives suffice. The independent pilot is root-$m$ regular with covariance $V$ and consistent $\widehat V$. Take $m/n\to\rho\in(0,\infty]$ and $\lambda_n=o(\sqrt n)$. On a null, positive definiteness of $S_j=\Omega_j+\rho^{-1}B_jVB_j^T$ is required.

Conditionally on the pilot, the regression expansion about $\beta(\theta_0,\widehat\theta)$ has Gaussian limit with covariance $\Omega_j$. Its conditional characteristic function tends in probability to a nonrandom one; multiplication by the bounded pilot characteristic function establishes limiting independence. A Taylor expansion of the target supplies $B_j(\widehat\theta-\theta_0)$ with root-$n$ remainder $O_{\Prob}(\sqrt n/m)=o_{\Prob}(1)$. Centering with a fitted intercept and the fixed ridge penalty contributes no root-$n$ bias. Therefore the null covariance is $S_j$. At $\rho=\infty$ pilot uncertainty vanishes, still using the assumed root-$m$ regularity. Empirical centered residual influences give consistent $\widehat\Omega_j$ and $\widehat B_j$; Slutsky's theorem yields the chi-square block test. The ordinary BY finite-threshold bound uses only marginal null convergence at its finitely many thresholds, not independence between p-values~\citep{by}. Variance addition is classical~\citep{murphy,newey}, and is not claimed as a new inference principle. Related learned-law studentization appears in \citet{niu}.

\subsection{Compatible pilot and local loading regularity}
Theorem~\ref{thm:pl} applies to the new star pilots with $\tau_m=1/m$, distinct from the previously documented default $m^{-2}$. Both satisfy $\sqrt m\tau_m\to0$. The Newton-polished gradient residual is bounded by $\min(10^{-8},m^{-2})$; stationarity is a floating-point diagnostic, not an interval certificate. The pilot sandwich uses the full subject gradient of the summed node losses. Cross-node score covariances are retained; opposite matrix parameters are transposes of the same fitted matrix. The experimental covariance uses the penalized Hessian, whose penalty tends to zero, and the empirical full subject score covariance. All source snapshots preserve this implementation.

For an edge with $\sigma_1(M_e)>\sigma_2(M_e)\geq0$ and $\sigma_1>0$, the operator-norm loading is smooth locally. The augmented $A_e$ has constant rank three, with its positive spectrum bounded away from zero locally. Its isolated zero-eigenspace projection $\Pi_e$ is smooth. The matrix
\[
 B_e=(A_e+\Pi_e)^{1/2}-\Pi_e
\]
is a smooth square root of $A_e$: it vanishes on the null space and equals the positive square root on the orthogonal complement. The Gaussian representation with independent standard latent vectors gives field derivatives with locally integrable polynomial Gaussian envelopes. Every fixed derivative of a bounded von Mises harmonic moment is a finite sum of bounded cumulants. Dominated differentiation therefore gives the needed first two moment derivatives, including residual moments by conditioning on $X$. The numerical eigenfactor and this factor induce the same conditional field law. The star's singular values $0.8$ and $0.5$ satisfy this premise. Zero edges, equal leading singular values and graph changes are outside this argument, even though the known-law kernel remains valid there.

\subsection{Leverage variant and computational cost}
Let $d_i^c=d_i-\bar d$, $G=n^{-1}\sum_i d_i^c(d_i^c)^T$ and
$h_i=n^{-1}+n^{-1}(d_i^c)^T(G+\lambda I/n)^{-1}d_i^c$. HC0 uses the empirical centered raw influences. The HC3-style variant uses uncentered outer products of
\[
 C_jG^{-1}d_i^c\,\widehat R_i/(1-h_i)
\]
for the selection variance, and leaves the raw cross-covariance $\widehat B_j$ unchanged. At $\lambda=0$ this is OLS HC3~\citep{hc3}; at positive ridge it is not an exact ridge deletion formula. With bounded fixed-dimensional design and positive definite population covariance, $\max_i h_i=o_{\Prob}(1)$, so both estimates are asymptotically equivalent. Their finite-sample behavior differs materially in the recorded global-null case.

With $r=4pH$, $d=2H$ and $k=2p+4|G|$, the recorded dense implementation costs
\[
 O(nr^2+r^3+nrk+pd^2k^2+pd^3)
\]
for regression, cross-covariance, its direct pilot covariance contraction and the block solves, excluding pilot fitting and generation. This includes the unoptimized three-factor contraction actually executed. Reordering it into two matrix products could reduce that component but would be an implementation change, not an executed speed improvement. Working storage includes the dense design and sufficient statistics, $O(n(r+k)+r^2+k^2+pdk)$, in addition to saved records. No linear generator-cost claim is extended to these statistics.

\section{Conditional-redraw information and sharpness}\label{app:latent_new}
For a null subset $N$, oracle swaps make $L_N$ centered with block diagonal covariance $\Omega_N$. Write $X^N$ for replacement of all its original null coordinates by witnesses, and $\Delta=\mathcal T(X)-\mathcal T(X^N)$. Simultaneous null swaps and the derivative identity give $\E L_N\Delta^T=-2B_N$. Under a conditional independent redraw, $\mathcal T(X)$ and $\mathcal T(X^N)$ are iid given $(Z,X_{-N})$, regardless of any remaining dependence with $Y$. Thus
\[
 \Var(\Delta)=2\E\Var\{\mathcal T(X)\mid Z,X_{-N}\}=2\mathcal I_N^{\rm res}.
\]
The positive semidefinite covariance of $(L_N,\Delta)$ gives the main article's half-information inequality by Schur complement. Whitening yields $B_N\mathcal I^{-1}B_N^T\preceq\Omega_N/2$. For any $V\succeq0$, $V\preceq\kappa_V\mathcal I^{-1}$, proving the variance inflation bound. The same proof applies to smooth scalar null-score influences; the common norm-score calculation is one such instance.

\subsection{An attained bound and a valid nonredraw control}
Consider the four-angle family $p_\theta(x)\propto\exp\{\theta\cos x_1\cos(2x_2)\}$ and fixed $Y=\cos(2X_2)$, at true parameter zero. Use latent $Z=\cos(2X_2)\in\{-1,1\}$ with equal prior probabilities. Conditional on $Z$, draw original and witness coordinate 1 independently with tilted mass proportional to $\exp\{\theta Z\cos x_1\}$, and draw original and witness coordinate 2 independently from the two angles having this value of $Z$. The two conditional normalizers for coordinate 1 coincide by antipodal symmetry, so the original marginal is exactly $p_\theta$. The joint law is invariant under both whole-coordinate swaps for all nearby $\theta$.

At zero, $\Gamma=I_8/2$, $\beta=0$, $\mathcal I=1/2$, $L_1=2\{v(X_1)-v(\widetilde X_1)\}Y$, $\Omega_1=4I_2$, and $B_1=(-1,0)^T$. Consequently $B_1^T\Omega_1^{-1}B_1=1/4=\mathcal I/2$. This proves sharpness over bounded finite-angle conditional-redraw families. It does not prove sharpness within the continuous bilinear family of the article.

For a contrasting valid kernel, keep the coordinate 2 redraw. Let $q_\theta$ be the conditional tilted law of coordinate 1 given $Z$ and $f_\theta(x)=(\cos x-\E_q\cos X)/\sqrt{\Var_q(\cos X)}$. Give its original/witness pair conditional mass
\[
 q_\theta(x)q_\theta(w)\{1-\gamma f_\theta(x)f_\theta(w)\},\qquad \gamma=1/4.
\]
It is nonnegative on a sufficiently small neighborhood of zero, symmetric, and has both $q_\theta$ marginals because $\E_q f_\theta=0$. Swapping coordinate 2 preserves $Z$, so all whole-coordinate swaps remain valid. The conditional cosine copy correlation at zero is $-\gamma$, and sine correlation is zero. The design is nonsingular with minimum eigenvalue $3/8$, $B_1=(-1,0)^T$, and $\Omega_1=\operatorname{diag}(4/(1+\gamma),4)$. Thus $B_1^T\Omega_1^{-1}B_1/\mathcal I=5/8$. A conditional independent identically distributed pair cannot have this negative covariance, so the kernel is outside the redraw class. The general information bound remains valid.

An independent script sums all 256 joint states without using project fitting or sampling routines. It checks all four swap sets at parameters $-0.1,0,0.1$, design eigenvalues, the information and influence matrices, and a separate finite difference of population regression coefficients. The largest swap error is $8.7\times10^{-19}$; derivative errors are below $5.3\times10^{-11}$. These floating-point sums verify the algebraic examples, not continuous von Mises sampling or FDR.

\subsection{Refinement of the earlier pilot-ratio calculation}
For the ten-pair efficient common pilot, the half-information theorem gives $c\leq1/20$, refining Proposition~\ref{prop:information}. The preceding ratio-six Corollary~\ref{cor:pilot_ratio} remains a valid general-information calculation. For this specific redraw kernel, its final bound strengthens to
\[
 F_\rho\leq F_0+\frac1{16\rho},\qquad \rho\geq3\ \Longrightarrow\ F_\rho\leq F_0+\frac1{48}<0.1.
\]
One may use its existing exact normal-density crossing calculation with $10c\leq1/2$. Equivalently, the derivative of a normal density in log variance has half-integrated absolute value $\E|Z_0^2-1|/4=\varphi(1)<1/4$, yielding total variation at most $\varphi(1)\log(1+10c/\rho)$. The FDP range is $[0,1/2]$. The exact comparison at ratio three is $591842767/5959489536<1/10$. This statement is asymptotic within Theorem~\ref{thm:shared}, not a finite-sample rule or a result at the unit-parameter numerical values.

\section{New star studies and their provenance}\label{app:new_studies}
The old 7000 pair datasets above remain distinct from the new 9000 star datasets. The initial five configurations have $p=12$, six signals, $n,m$ equal to $400,400$ or $1600,400$, plus a global-null case and a $1600,40$ pilot stress case. A separate $n=32000,m=400$ configuration has 1000 new repetitions. The subsequent ten-signal plan has $p=20$ and three configurations: continuous $1600,400$, continuous $1600,1600$, and binary $1600,400$, each with 1000 new repetitions. Exact star rejection sampling, general fields and matrices, compatible pilot fitting and fixed response parameters are specified in the main article and frozen configuration files.

Matched sensitivity versions apply $H=1,2$ and HC0 or HC3-style to the same 5000 initial and 3000 ten-signal datasets, respectively. They add no independent repetitions. Their 16 methods include pilot-adjusted, unadjusted and oracle Wald/BY tests, and fitted-law or oracle knockoff+ at each basis size. All methods use nominal $q=0.1$. The ten-signal principal choice $H=2$/HC3-style was recorded before its new outcomes, following the earlier discovery-floor finding. It is not an outcome-tuned choice for individual datasets or a blind confirmation of the original plan.

An independent audit reconstructs the chi-square tails without importing the selector, uses separate linear algebra and threshold code, and recomputes 128000 decisions over the 8000 matched datasets. Its 128 aggregate rows and 2880 paired contrasts agree with the retained outputs. The source CSV and its hash identify every displayed new numerical row. All 9000 new runs and both matched reanalyses completed with zero recorded failures. The designated replays number 15, 3 and 9 for the initial, large-$n$ and ten-signal generation, plus 15 and 9 for their matched reanalyses. These 51 checks reuse seeds and are not independent experiments.

The large-$n$ six-signal comparison gives adjusted Wald power one but fitted-law knockoff+ power $0.056$, consistent with its ordinary discovery floor; it does not establish a better procedure. In the new binary ten-signal comparison, pilot correction reduces power substantially while both uncorrected and corrected FDR remain below nominal. The global-null $H=2$ known-law HC0 Wald FDR is $0.193$ (MCSE $0.0125$), versus $0.007$ ($0.0026$) for the HC3-style version. This finite-sample problem is retained, not explained away as covariate estimation error. No general finite-sample or high-dimensional robustness claim is made.
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bibliography(text: str) -> str:
    import re
    old = text.split(r'\begin{thebibliography}{99}', 1)[1].split(r'\end{thebibliography}', 1)[0]
    entries = re.findall(r'\\bibitem.*?(?=\\bibitem|\Z)', old + NEW_BIB, flags=re.S)
    entries.sort(key=lambda entry: entry.split(']{', 1)[0].lower())
    return '\n'.join(entry.strip() + '\n' for entry in entries)


def tables(rows: list[dict[str, str]]) -> tuple[str, str]:
    def one(version: str, case: str, method: str) -> dict[str, str]:
        found = [r for r in rows if (r['version'], r['configuration'], r['method']) == (version, case, method)]
        if len(found) != 1 or found[0]['repetitions'] != '1000':
            raise RuntimeError('Missing, duplicated or wrong-size numerical source')
        return found[0]
    def number(row: dict[str, str], key: str) -> str:
        return f"{float(row[key]):.4f} ({float(row[key + '_mcse']):.4f})"
    null = r"""\begin{table}[htbp]\centering\small
\caption{Global-null star sensitivity: $n=m=400$, $p=12$, $H=2$. Each row uses the same 1000 datasets. FDR equals the probability of any selection here. The fitted-law rows use a joint compatible pilot; the known-law rows have no estimated pilot. Parentheses give MCSEs.}\label{tab:null}
\begin{tabular}{llr}\toprule
Wald/BY variance & Covariate law & FDR\\\midrule
"""
    for hc, label in [('hc0', 'HC0'), ('hc3style', 'HC3-style')]:
        for method, kind in [('pilot', 'Fitted; pilot variance'), ('naive', 'Fitted; no pilot variance'), ('oracle', 'Known')]:
            row = one('sp06_3', 'null_n400_m400', f'h2_{hc}_wald_{method}_by')
            null += f"{label} & {kind} & {number(row, 'fdp')} \\\\\n"
    null += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    fair = r"""\begin{table}[htbp]\centering\small
\caption{Ten-signal star comparison: $p=20$, $n=1600$, $H=2$, nominal $q=0.1$, 1000 repetitions per configuration. All Wald rows use the prespecified HC3-style variance and BY. Power is unconditional; Empty is the probability of no selection. Parentheses give MCSEs. Methods are paired within each configuration.}\label{tab:fair}
\begin{tabular}{lrrr}\toprule
Method & FDR & Power & Empty\\\midrule
"""
    names = [('h2_hc3style_wald_pilot_by', 'Wald + pilot variance'),
             ('h2_hc3style_wald_naive_by', 'Wald, no pilot variance'),
             ('h2_hc3style_wald_oracle_by', 'Known-law Wald'),
             ('h2_knockoff_plus_pilot', 'Fitted-law knockoff+'),
             ('h2_knockoff_plus_oracle', 'Known-law knockoff+')]
    for case, label in [('harmonic_n1600_m400_p20', r'Continuous, $m=400$'),
                        ('harmonic_n1600_m1600_p20', r'Continuous, $m=1600$'),
                        ('binary_n1600_m400_p20', r'Binary, $m=400$')]:
        fair += r'\multicolumn{4}{l}{' + label + '}' + ' ' + chr(92)*2 + chr(10)
        for method, name in names:
            row = one('sp06_4_hc', case, method)
            fair += name + ' & ' + ' & '.join(number(row, key) for key in ('fdp', 'power', 'empty')) + ' ' + chr(92)*2 + chr(10)
        fair += '\\addlinespace\n'
    fair += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    return null, fair


def main() -> None:
    manuscript = ROOT / 'manuscript/obstacles_v3.tex'
    supplement = ROOT / 'manuscript/obstacles_supplement_spl01.tex'
    for path, expected in [(manuscript, '8c4290eec4c8a80fed264b54daaa141cf58c09afe6373ba8e8a426d68de73e14'),
                           (supplement, '6d84262b75f2bcfe89cbebe20a61abc887cd3a2adbf147087bbd4809aad213ea')]:
        if digest(path) != expected:
            raise RuntimeError('Source changed; do not overwrite: ' + str(path))
    if json.loads((AUDIT / 'receipt.json').read_text())['status'] != 'complete':
        raise RuntimeError('Numerical audit is incomplete')
    with (AUDIT / 'summary_recomputed.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 128:
        raise RuntimeError('Unexpected numerical row count')
    original = manuscript.read_text()
    old_supp = supplement.read_text()
    null, fair = tables(rows)
    preamble = original.split(r'\begin{abstract}', 1)[0]
    preamble = preamble.replace('% SPL revision spl01. Previous full source preserved in provenance.',
                                '% Statistical Papers research revision SP08. Earlier SPL source preserved immutably.')
    preamble = preamble.replace(r'\usepackage{amsmath,amssymb,amsthm}', r'\usepackage{amsmath,amssymb,amsthm,booktabs,array}')
    preamble = preamble.replace(r'\newtheorem{theorem}{Theorem}', r'\newtheorem{theorem}{Theorem}' + '\n' + r'\newtheorem{corollary}[theorem]{Corollary}')
    original_core = original.split(r'\section{Angular blocks and witnesses}', 1)[1].split(r'\section{Scope and numerical evidence}', 1)[0]
    original_core = r'\section{Angular blocks and witnesses}' + original_core
    # Berti's full published Theorem 12 and Cohen's augmentation were read in this revision.
    original_core = original_core.replace(r'framework of \citet{dreassi}', r'framework of \citet{berti,dreassi}')
    article = (preamble + '\\begin{abstract}\n' + ABSTRACT + '\n\\end{abstract}\n'
               + r'\noindent Keywords: knockoffs; angular covariates; false discovery rate; pilot estimation; latent augmentation.' + '\\\\\n'
               + 'MSC 2020: 62H15; 62F05; 62E20.\n\n' + INTRO + original_core + WALD + INFORMATION
               + NUMERICAL_START + null + NUMERICAL_END + fair + DISCUSSION
               + '\n\\begin{thebibliography}{99}\n' + bibliography(old_supp)
               + '\\end{thebibliography}\n\\end{document}\n')
    old_supp = old_supp.replace('This file contains the complete theoretical and numerical development supporting the short article. Its results are numbered S1--S11. Main Theorems 1 and 2 correspond to Theorems S2 and S4 here. All four tables, seven toroidal configurations, paired comparisons and the separate Gaussian-limit calculation are retained. No numerical study has been replaced by the shortening of the main article.',
                               'This supplementary document retains the original results S1--S11, four tables and seven pair-study configurations, and adds the independent-pilot block-test details, conditional-redraw information calculation and general-matrix star provenance. Main Theorems 1 and 2 still correspond to Theorems S2 and S4. The new main results are proved in the article and developed in the final appendices here. Historical experimental versions remain separate.')
    old_supp = old_supp.replace('Its Gaussian limit gives a conservative sufficient pilot-to-selection ratio of six within the same counterexample.',
                               'Its Gaussian limit first gives a conservative sufficient pilot-to-selection ratio of six within the same counterexample. The conditional-redraw refinement in the final appendices improves that sufficient ratio to three, without changing the experiments.')
    old_supp = old_supp.replace('OpenAI Codex and Anthropic Claude assisted with mathematical arguments, computational code and editing. The author reviewed the proofs, references and results. Verification distinguishes executed tests and seed replays from reaggregation of retained simulations, as documented in Appendix J.',
                               'OpenAI Codex and Anthropic Claude assisted with mathematical arguments, computational code and editing. The author reviewed the earlier material; the new revisions remain a draft for final author review. Verification distinguishes executed tests and seed replays from reaggregation of retained simulations.')
    before_bib = old_supp.split(r'\begin{thebibliography}{99}', 1)[0]
    appended = before_bib + SUPPLEMENT + '\n\\begin{thebibliography}{99}\n' + bibliography(old_supp) + '\\end{thebibliography}\n\\end{document}\n'
    # Every new cited item is also introduced in the supplementary discussion.
    if '\u2014' in article or '\u2014' in appended:
        raise RuntimeError('Em dash introduced')
    manuscript.write_text(article)
    supplement.write_text(appended)
    record = {'status': 'integrated_not_compiled', 'version': 'sp08',
              'source_csv': str((AUDIT / 'summary_recomputed.csv').relative_to(ROOT)),
              'source_csv_sha256': digest(AUDIT / 'summary_recomputed.csv'),
              'manuscript_sha256': digest(manuscript), 'supplement_sha256': digest(supplement),
              'builder_sha256': digest(Path(__file__)), 'new_simulations': 0,
              'author_review': 'pending for SP08 revisions'}
    output = ROOT / 'results/statistical_papers/sp08_integration_20261010'
    output.mkdir(exist_ok=False)
    (output / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
