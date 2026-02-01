# Constants
n: number of observations
K: number of regressors in the (unrestricted) model
J : number of restrictions
m: number of instruments (or discrepancy vector for restrictions)

#  Formulas
- Expected value $E[X] = \sum_{i}x_iP(X=X_i)$ or $E[X]=\int_{-\infty}^\infty x\cdot f_X(x)dx$
- Variance $\text{Var}[x]=E[(X-\mu_X)^2] = E[X^2]-E[X]^2$
- Jensen's inequality: If g convex, then $g(E[X])\leq E[g(x)]$
- Expectation inequality: $|E[X]| \leq E[|X|]$
- Cauchy-Schwarz inequality: $|E[XY]|\leq \sqrt{E[X^2]E[Y^2]}$
- rth raw moment: $E[X^r]$, rth central moment: $E[(X-E[X])^r]$
- Skewness: $\frac{E[(X-\mu_X)^3]}{\sigma_X^3}$. Measure of symmetry, if skewness = 0 symmetric
- Kurtosis: $\frac{E[(X-\mu_X)^4]}{\sigma_X^4}$ Measure of tail mass. If kurtosis > 3 heavy-tailed
- Conditional distribution: $P(Y=y|X=x) = \frac{P(X=x, Y=y)}{P(X=x)}$ (discrete) $f_{Y|X=x}(y) = \frac{f_{X,Y}(x,y)}{f_X(x)}$ (continuous)
- Simple Law of iterated expectations: $E[Y]=E[E[Y|X]]$
- Extended law of iterated expectations: $E[Y|X]=E[E[Y|X,Z]|X]$
- Variance decomposition: $\text{Var}[Y] = E[\text{Var}[Y|X]]+ \text{Var}[E[Y|X]]$
- Independance: $P(Y=y|X=x) = P(Y=y)$ or $f_{Y|X=x}(y)=f_Y(y)$
- Covariance: $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] = E[XY]-E[X]E[Y]$
- Correlation: $\text{Corr}(X,Y)=\frac{\text{Cov}(XY)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$
- $Var(AX) = AVar(X)A', Cov(AX,Y) = ACov(X,Y), Var(X\pm Y) = Var(X)+Var(Y)\pm 2Cov(X,Y)$

# CLT, LLN
Let $X_i$ be i.i.d., $E[X_i]=\mu, Var(X_i) = \sigma^2$.
$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \overset{p}{\rightarrow} \mu$$
$$\sqrt{n}(\bar X_n-\mu)\overset{d}{\rightarrow}\mathcal{N}(0,\sigma^2), \frac{\bar X_n-\mu}{\sigma/\sqrt{n}}$$

# Distributions
t-test large sample critical values at 5% significance level: 1.96
If $z\in\mathbb{R}^{J+1}, z\sim\mathcal{N}(\mu,\Sigma)$, then $(z-\mu)'\Sigma(z-\mu)\sim\chi^2(J)$

# Test single OLS coefficients
**Null hypothesis:** $H_0 :\beta_k = \beta_{k,0}$
**Test statistic:** t-statistic $$t_k = (b_k - \beta_{k,0})/ \sqrt{s^2(X'X)_{kk}^{-1}} \sim t(n-K)$$
with $s^2 = \frac{e'e}{n-K}$. Usually we are given $se(b_k)=\sqrt{s^2(X'X)_{kk}^{-1}}$

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
**Method:** Regress $e_i^2$ on $z = \left(1, x_{i1},...,x_{ik}, x_{i1}^2, ..., x_{ik}^2, x_{i1}x_{i2}, ..., x_{i,k-1}x_{ik}\right)$ and take $R^2$ of the regression
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

Check if $\beta_i = \beta_j = 0$. Have unrestricted regression with $R^2_u$ and restricted with $R^2_r$.
J = number of restrictions
n = number of observations
K = number of regressors in the unrestricted model
$F = \frac{(R^2_u-R^2_r)/J}{(1-R^2_u)/(n-K)} \sim F(J, n-K)$
Look at quantiles of this distribution


# OLS estimator
Two regressors:
$$b = (X'X)^{-1}X'y$$
$$b_2 = \frac{\sum_{i=1}^n (x_{i2}- \bar{x}_{i2})(y_i-\bar y)}{\sum_{i=1}^n (x_{i2}-\bar x_2)^2} = \frac{S_{x_2y}}{S_{x_2}^2}$$
$$b_1 = \bar y - b_2\bar x_2$$

# Two stage least squares
$$b^{TSLS} = (X'\underbrace{Z(Z'Z)^{−1}Z'}_{P_Z(X)}X)^{−1}X'Z(Z'Z)^{−1}Z'y$$
$$b_2^{TSLS} = \frac{S_{\widehat{x_2}y}}{S_{\widehat x_2}^2} = \frac{\widehat\pi_2 S_{z_2y}}{\widehat\pi_2^2 S_{z_2}^2}\overset{\widehat\pi_2 \text{OLS of z on x}}{=}\frac{S_{x_2}^2}{S_{x_2 z}}\frac{S_{z_2y}}{S_{z_2}^2} = \frac{S_{z_2y}}{S_{x_2z}}$$

Often: eliminate $\bar\epsilon$: $$\sum_i (z_{i2}-\bar z_2)(\epsilon_i-\bar\epsilon) = \sum_i (z_{i2}-\bar z_2)\epsilon_i-\sum_i(z_{i2}-\bar z_2)\bar\epsilon = \sum_i (z_{i2}-\bar z_2)\epsilon_i-\bar\epsilon\underbrace{\sum_i(z_{i2}-\bar z_2)}_{=0}$$

# Panel data estimators
$$b^{POLS} = \left( \sum_i\sum_t (x_{it}-\bar{\bar x})(x_{it}-\bar{\bar x})'\right)^{-1} \left( \sum_i\sum_t (x_{it}-\bar{\bar x})(y_{it}-\bar{\bar y})\right)$$

$$b^{FE} = \left( \sum_i\sum_t (x_{it}-\bar{x_i})(x_{it}-\bar{x_i})'\right)^{-1} \left( \sum_i\sum_t (x_{it}-\bar{x_i})(y_{it}-\bar{y_i})\right)$$

# Omitted variable bias
Two models $y_i = \beta_1 + \beta_2x_{i2} + \epsilon_i$ and $y_i = \gamma_1 + \gamma_2x_{i2} + \gamma_3x_{i3} + u_i$.
Then
$\beta_2 = \gamma_2 + \gamma_3\frac{Cov(x_{i2},x_{i3})}{Var(x_{i2})}$

# Keep in mind
Check finite moments
Write down assumptions (CLT, CMT, LLN)