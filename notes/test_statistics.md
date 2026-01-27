# Constants
n: number of observations
K: number of regressors in the (unrestricted) model
J : number of restrictions
m: number of instruments (or discrepancy vector for restrictions)


# Large sample critical values at 5% significance level
t-test: 1.96



# Test single OLS coefficients
**Null hypothesis:** $H_0 :\beta_k = \beta_{k,0}$
**Test statistic:** t-statistic $$t_k = (b_k - \beta_{k,0})/ \sqrt{s^2(X'X)_{kk}^{-1}} \sim t(n-K)$$
with $s^2 = \frac{e'e}{n-K}$

# Test restrictions
**Regression model:** $y = X\beta+\epsilon$
**linear restrictions:** $R\beta=q, R\in\mathbb{R}^{J\times K}$
**Null hypothesis:** $H_0: R\beta-q = 0, \quad H_1: R\beta-q \neq 0$
**Distribution under $H_0$:** Discrepancy vector $m = Rb − q$. $m|X \sim \mathcal{N}(0, \sigma^2R(X'X)^{-1}R')$
Wald criterion $$W = m'(Var[m|X])^{−1} m = (Rb − q)'[\sigma^2R(X'X)^{−1}R']^{−1}(Rb − q) \sim \chi^2(J)$$
We do not observe $\sigma^2 \rightarrow$ use $s^2$ and obtain F-statistic
**Test statistic:** $$F = \frac{(Rb − q)'[s^2R(X'X)^{−1}R']^{−1}(Rb − q)}{J} \sim F(J, n-K)$$
For one restriction $F = t^2$

# Test for heteroskedasticity (White's test)
**Null hypothesis:** $H_0: \sigma_i^2 = \sigma^2$
**Method:** Regress $e_i$ on $z = \left(1, x_{i1},...,x_{ik}, x_{i1}^2, ..., x_{ik}^2, x_{i1}x_{i2}, ..., x_{i,k-1}x_{ik}\right)$ and take $R^2$ of the regression
**Test statistic:** $nR^2 \xrightarrow{d}\chi^2_{J-1}$ under $H_0$. Reject if $nR^2 > \chi_{J−1,1−\alpha}​$

# Test instrument validity
K: number of endogenous regressors
r: number of exogenous regressors
## IV regression
$$y_i =\beta_0 +\beta_1x_{i1} +...+\beta_K x_{iK} +\beta_{K+1}w_{i1} +...+\beta_{K+r}w_{ir} +\epsilon_i$$
## 1. Relevance: $Cov(z_i, x_i) \neq 0$ 
**Method:** First stage regression: $$x_{ik} = \pi_{0,k} + \pi_{1,k}z_{i1}+ ... + \pi_{m,k}z_{im} + \pi_{m+1, k}w_{i1} + ... + \pi_{m+r, k}w_{ir} + \nu_i$$ $$\rightarrow \widehat{x_{1k}}, ..., \widehat{x_{nk}}$$
**Null hypothesis:** $$\pi_{1,k} = ... = \pi_{m,k} = 0$$ 
**Test statistic:** Same as testing for restrictions: Define $\widehat{\Pi} = \left(\pi_{0,k}, ..., \pi_{m,k}\right)^T $
$F = \frac{\widehat{\Pi}'(Z'Z/s^2)\widehat{\Pi}}{m} \sim F(m, n-m)$
**Rule of thumb:** $K=1, F>10$

## 2. Exogeneity: we need that $m>K$ (more instruments than endogenous regressors)
**Method:** Second stage regression: Regress $y_i$ on $\widehat{x_{1k}}, ..., \widehat{x_{nk}}, w_{i1}, ..., w_{ir}$
$\rightarrow b^{TSLS} = \left(b_0^{TSLS}, ..., b_{K+r}^{TSLS} \right)^T$
Use true $x_{ik}$ to calculate 
$$\widehat{\epsilon}_i^{TSLS} = y_i - (b_0^{TSLS} + b_1^{TSLS}x_{i1} + ... + b_{K+r}^{TSLS}w_{ir})$$.
Regress $\widehat{\epsilon}_i$ on $z_{i1},...,z_{im}, w_{i1}, ..., w_{ir}$
**Null hypothesis:** $H_0:$ The coefficients for $z_{i1},...,z_{im}$ in the regression of $\widehat{\epsilon}_i^{TSLS}$ are 0
**Test statistic** Again F-statistic. In large samples $J = mF \sim \chi^2(m − K)$


# Testing Random vs. Fixed Effects: Hausman Test
**Null hypothesis:** $\text{Cov}(z_i, x_{it}) = 0$
**Method:** Calculate Fixed Effects estimate $b^{FE}$ and Random Effects estimate $b^{GLS}$
**Test statistic:** $$W=(b^{FE}-b^{GLS})'(\widehat{Var}(b^{FE})- \widehat{Var}(b^{GLS}))^{-1}(b^{FE}-b^{GLS}) \sim \chi^2(K),$$