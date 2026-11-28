# Estimated-scale reference enclosures — issue #5

This candidate adds a deterministic reference component, not a new runtime
precision admission. It leaves PR #39's exact-record, fixed-scale censor and
reflection decisions unchanged. No change to the precision dispatcher, CLI,
random seeds, scientific tolerances, or CI policy.

## Mathematical construction

Let theta=(mu, eta), sigma=exp(eta). The supplied prior is N(m,P) in
these coordinates, including the covariance off-diagonal. For n positive
exact records, the lognormal density contributes exp(-n eta) times
constant observed-discharge factors and the product of
exp(-((log(q/reference)-mu)/sigma)^2/2). Each classified censor probability
is a factor in [0,1]. Excluded/missing observations have no factor.

Completing the Gaussian square absorbs exp(-n eta) into a tilted Gaussian
with mean m* = m - n P e_eta and unchanged covariance P. The remaining
likelihood W is in [0,1], and the normalized posterior is proportional to
the tilted Gaussian density times W. Its normalizer is Z=E*[W], not the
original marginal likelihood including discarded constants.

Use exact rational SPD and determinant checks on stored binary64 values,
and outward Decimal arithmetic from the accepted fixed-scale reference.
Whiten with independent standard Normals z0,z1 using the lower Cholesky
factor of P. On each dyadic cell, enclose both coordinates, sigma, every
exact residual and censor probability. The cell Gaussian probability is
the product of two enclosed one-dimensional masses. The integral of
W h on that cell is enclosed by the probability mass multiplied by the
interval range of W h. Sign-changing h is handled by interval products.
The finite grid covers a square [-a,a]^2 in independent coordinates.

Omitted Gaussian mass T is bounded by min(1,4 SF(a)). Since W<=1, the
omitted normalizer is in [0,T], and any probability/indicator numerator
has the same omitted bound. For coordinate means, exponentials of either
coordinate, and centered coordinate second moments, use Cauchy–Schwarz:

    E*[|h| 1_outside W] <= sqrt(E*[h^2] T).

For a tilted marginal N(M,V), the required second moments are
M^2+V for h=X, exp(2M+2V) for h=exp(X), and
(M-c)^4+6(M-c)^2 V+3V^2 for h=(X-c)^2. Include a signed tail
interval for coordinate means and a nonnegative interval otherwise.
Normalize only when the lower bound on Z is strictly positive.
The result concerns the full Gaussian target, not a truncated prior.

Supported scalar reference requests: both coordinate means, both
exponential coordinate means, both centered coordinate second moments,
direct expected F and AEP at fixed positive discharge, and mu CDF/SF at
a declared threshold. The output retains model inputs, fixed grid controls,
reserved cell terms, tilted means, normalizer and omitted Gaussian mass
enclosures. Cell/term caps and duplicate requests are checked before
the reference calculation. Output arithmetic failure retains the reason
instead of yielding an invented value.

## Independent numerical checks

Synthetic correlated case: prior mean (0,0), covariance
((0.16,0.02),(0.02,0.0144)); reference discharge 1; exact log flows
(-0.4,0.5), left censor -0.2, right censor 0.1, interval (-0.5,0.5).
An independent SciPy/NumPy Gauss-Legendre calculation integrated over
[-10,10] in each independent Gaussian coordinate, splitting the first
axis at mu=0. Orders 64, 96, 128 agreed to about 6e-15 absolute across
the normalizer and ten integrands. This is a numerical cross-check,
not the omitted-tail proof.

| Quantity | Independent numerical reference |
|---|---:|
| Tilted normalizer | 0.0455818189049154 |
| E(mu) | -0.00827612383713244 |
| E(eta) | -0.0222234418867801 |
| E(exp(mu)) | 1.03928232666411 |
| E(exp(eta)) | 0.984394213479628 |
| E(mu squared) | 0.0928259626422699 |
| E(eta squared) | 0.0134669252021155 |
| Expected F at exp(0.35) | 0.640717343006427 |
| Direct expected AEP at exp(0.35) | 0.359282656993573 |
| P(mu <= 0) | 0.516776416243289 |
| P(mu > 0) | 0.483223583756711 |

With extent 4, 64 versus 1024 cells, all ten enclosures narrowed and
contained the independent references. The normalizer narrowed from
[0.01118965,0.11382479] to approximately [0.03400252,0.05944271].
The 1024-cell expected AEP enclosure is still broad, approximately
[0.19443837,0.66457238]. Do not claim that this grid supplies useful
engineering precision. The omitted Gaussian mass upper bound is
0.000126685. The coarse/fine cell-term reservations are 1192 and 16600.

Four focused pytest tests passed in 7.77 seconds on local Python 3.13.5
against a locally transcribed excerpt of the accepted interval arithmetic,
not a repository checkout. They exercise the correlated mixed case,
reflected analytic identities, deliberately large omitted tails, ambient
Decimal context isolation, exact-only unit scaling and direct rare AEP,
invalid covariances, duplicate requests, and both resource caps.
The independent quadrature and this focused harness do not replace
testing the complete repository on Python 3.11 and 3.14.

## Required next checks

Review the finite-cell and tail argument independently; run the committed
tests with the repository's actual interval arithmetic and all pinned
quality tools. Assess tighter partitions or analytic conditioning without
silently changing a fixed budget. Add typed model/proposal identity
binding and reuse the accepted ratio envelope and moment eligibility
before integrating any additional runtime pass. General inverse endpoints
for design quantiles, two-coordinate transforms and installed precision
settings remain open. In particular do not evaluate a divergent upper
design mean merely because the finite box has finite values.

The shared method registry is deliberately unchanged during concurrent
integration #40; update it with the reviewed implementation and limits
before marking the PR ready. Keep #5 open; no merge, native qualification,
publication, or release is authorized.
