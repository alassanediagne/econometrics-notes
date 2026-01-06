
#  Prep Course

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
- Correlation: $\text{Corr}(X,Y)=\frac{\text{Var}(XY)}{\text{Var}(X)\text{Var}(Y)}$

# Chapter 1: The Linear Regression Model

## Linear Regression
- Causal effect $E[Y|X=1] - E[Y|X=0]$ 
- Total Sum of Squares $TSS = \sum_{i=1}^n (y_i-\bar y)^2$
- Explained Sum of Squares $ESS = \sum_{i=1}^n (\hat y_i-\bar y)^2$
- Sum of Squared Residuals $SSR = \sum_{i=1}^n e_i^2$
- Coefficient of Determination 
  $$R^2 = \frac{ESS}{TSS} = 1-\frac{SSR}{TSS} = \widehat{corr}(y_i, x_{i2})^2$$ 
  $\rightarrow$ high $R^2$ is good
- Standard error of the regression (Stata: Root MSE) 
  $$s_e = \sqrt{ \frac{1}{n-2} \sum_{i=1}^n e_i} = \sqrt{\frac{SSR}{n-2}}$$

### OLS Assumptions
**Least Squares Assumptions for Prediction:**
$(x_i, y_i)$ in sample data used to estimate regression coefficients, $(x_i^{OOS}, y_i^{OOS})$ out of sample data
$$E[y|x_2] = \beta_1 + \beta_2x_2,\quad \epsilon = y- E[y|x_2]$$
- **(A1)** $(x_i^{OOS}, y_i^{OOS})$ are randomly drawn from the same population distribution as $(x_i, y_i)$
- **(A2)** $(x_i, y_i)$ are i.i.d. draws from their joint distribution
- **(A3)** $(x_i, \epsilon_i)$ have nonzero finite fourth moments

**Least Squares Aussumptions for Causal Inference**
structural model $y_i = \beta_1 + \beta_2 x_{i2} + \epsilon_i$
- **(A1)** Conditional mean zero: $E[\epsilon_i|x_{i2}] = 0$
- **(A2)** $(x_{i2}, y_i)$ are i.i.d. draws from their joint distribution
- **(A3)** $(x_{i2}, \epsilon_i)$ have nonzero finite fourth moments
- **(A4)** Homoskedasticity: $\text{Var}[\epsilon_i|x_{i2}] = \sigma_\epsilon^2$
- **(A5)** Conditional normality: $\epsilon_i|x_{i2} \sim \mathcal{N}(0,\sigma_\epsilon^2)$

Prediction:
- Goal: get Y right
- High $R^2$ / low error = good
- A biased model can still predict well

Causal inference:
- Goal: estimate “what happens if we intervene”
- Bias = fatal
- A model with terrible $R^2$ can still give correct causal effects

### Treatment effects
- Treatment effect $te_i = Y_i(1)-Y_i(0)$ (but: we usually cannot observe both)
- Average Treatment Effect (ATE/ACE): $\tau_{ATE}=E[te_i] = E[Y_i(1)-Y_i(0)] = E[Y_i(1)]-E[Y_i(0)]$
- Average Treatment Effect on the Treated (ATET): $\tau_{ATET} = E[Y_i(1) | X_i=1] = E[Y_i(0) | X_i=1]$
- Average Treatment Effect on the Non-Treated (ATENT): $\tau_{ATENT} = E[Y_i(1) | X_i=0] = E[Y_i(0) | X_i=0]$
- ATE = $\pi\tau_{ATET} + (1 − \pi) \tau_{ATENT}$
  $$ E[Y_i|X_i = 1] − E[Y_i|X_i = 0]\\ = \tau_{ATE} + \underbrace{E[Y_i(0)|X_i=1]-E[Y_i(0)|X_i=0]}_{\text{selection bias}}\\ + \underbrace{(1 − \pi) (\tau_{ATET}-\tau_{ATENT})}_{\text{heterogeneous treatment effect bias}} $$ 
- $\rightarrow$ Randomized controlled experiment (RCT): random entities and random assignments
- Under random assignment selection bias and heterogeneous treatment effect bias = 0 $\rightarrow$ $E[Y_i|X_i = 1] − E[Y_i|X_i = 0] =\tau_{ATE}$
- relation to linear reg.: $\beta_1 = E[Y_i(0)], \beta_2 = \tau, \epsilon_i = \epsilon_i(0)$

### Multiple linear regression

multiple linear regression: $y=X\beta+\epsilon$

**Least Squares Aussumptions for Causal Inference (multi dimensional)**
structural model $y_i = \beta_1 + \beta_2 x_{i2} + \epsilon_i$
- **(A1)** Conditional mean zero: $E[\epsilon_i|x_{ik}] = 0, \quad k=1,...,K$
- **(A2)** $(x_{i2}, y_i)$ are i.i.d. draws from their joint distribution
- **(A3)** $x_{i}, \epsilon_i$ have nonzero finite fourth moments
- **(A4)** Full rank: $\text{rank}(X) = K$
- **(A5)** Conditional homoskedasticity: $\text{Var}[\epsilon_i|x_i] = E[\epsilon_i^2|x_i]= \sigma^2$
- **(A6)** Conditional normality: $\epsilon_i|x_{i2} \sim \mathcal{N}(0,\sigma_\epsilon^2)$

$\rightarrow$ Least squares: 
  $$\min_{b_0} e_0'e_0 = (y-Xb_0)'(y-Xb_0)$$
- b satisifies normal equations $(X'X)b = X'y$ $\rightarrow$ $b = (X'X)^{-1}X'y$
- Adjusted $R^2$: $R^2$ increases with $K$ $\rightarrow$ $\bar R^2 = 1-\frac{n-1}{n-K}(1-R^2)$

Estimatior $s^2$ of $\sigma^2$: $s^2 = \frac{e'e}{n-K}$

## Statistical Properties
### Finite sample properties
1. $E[b|X] = E[b] = \beta$
2. $\text{Var}[b|X] = \sigma^2(X'X)^{-1}$
3. Gauss-Markov Theorem: In the classical linear regression model with regressor matrix $X$, the least squares estimator $b$ is efficient in the class of linear (conditionally) unbiased estimators, i.e. for any linear (conditionally) unbiased estimator $b_0$ for $\beta$, $\text{Var}[b_0|X] - \text{Var}[b|X]$ is positive semidefinite
4. $E[s^2|X] = E[s^2] = \sigma^2$
Additionally, under assumption (A6)
5. $b|X \sim \mathcal{N}(\beta, \sigma^2(X'X)^{-1})$
6. $(n−K)s^2/\sigma^2 \sim \chi^2(n−K)$
7. For test $H_0 :\beta_k = \beta_{k,0}$ use t-statistic $$t_k = (b_k - \beta_{k,0})/ \sqrt{s^2(X'X)_{kk}^{-1}} \sim t(n-K)$$

### Large sample properties
**(A4a)** $E[x_ix_i'] = Q$ is positive definite. Do not assume **(A6)**
- $$b = \beta + (X'X)^{-1}(X'\epsilon) = \beta + \left(\frac{1}{n}\sum_{i=1}^n x_ix_i'\right)^{-1}\left(\frac{1}{n}\sum_{i=1}^n x_i\epsilon_i\right) \xrightarrow{P} \beta$$
- $\sqrt{n}(b-\beta) \xrightarrow{d} \mathcal{N}(0, Q^{-1}\Sigma Q^{-1})$
- Thus under **(A5)**$$\sqrt{n}(b-\beta) \xrightarrow{d} \mathcal{N}(0, \sigma^2Q^{-1}) \quad \Rightarrow \quad b \xrightarrow{d}\mathcal{N}(\beta, \frac{\sigma^2}{n}Q^{-1})$$
and $AVar[b] = \frac{\sigma^2}{n}Q^{-1}, \quad\widehat{AVar}[b]=s^2(X'X)^{-1}$

**Testing coefficients**
$H_0 :\beta_k = \beta_{k,0}$
Under the null hypothesis $$\frac{\beta_k - \beta_{k,0}}{AVar[b_k]}\overset{a}{\sim}\mathcal{N}(0,1)\quad \Rightarrow\quad \frac{\beta_k - \beta_{k,0}}{\widehat{AVar}[b_k]}\overset{a}{\sim}\mathcal{N}(0,1)$$

**Control variables**
If we have observational data with causal factors we cannot observe, introduce control variables which are correlated with these omitted causal factors but are not causal themselves
$y_i =\beta_1 + \beta_2x_{i2} +\beta_3W_i + \epsilon_i$
New **(A1)** conditional mean
independence:  $E[\epsilon | x_{i2}, W_i] = E[\epsilon | W_i]$

What makes a good control variable?
- Makes the error term uncorrelated with the variable of interest.
- Holding the control variable(s) constant, the variable of interest is “as if” randomly assigned.
- Among individuals with the same value of the control variable(s), the variable of interest is uncorrelated with the omitted determinants of y_i .

## Inference
Regression model $y = X\beta+\epsilon$, linear restrictions $R\beta=q, R\in\mathbb{R}^{J\times K}$
Example: $\beta_1=\beta_2 \rightarrow r_{11}=1, r_{12}=-1, q_1=0$
Assume **(A6)** 
Hypothesis $H_0: R\beta-q = 0, \quad H_1: R\beta-q \neq 0$
Discrepancy vector $m = Rb − q$. Under $H_0$: $m|X \sim \mathcal{N}(0, \sigma^2R(X'X)^{-1}R')$
Wald criterion $$W = m'(Var[m|X])^{−1} m = (Rb − q)'[\sigma^2R(X'X)^{−1}R']^{−1}(Rb − q) \sim \chi^2(J)$$
We do not observe $\sigma^2 \rightarrow$ use $s^2$ and obtain F-statistic
$$F = \frac{(Rb − q)'[s^2R(X'X)^{−1}R']^{−1}(Rb − q)}{J} \sim F(J, n-K)$$
For one restriction $F = t^2$
In large samples we do not need **(A6)**

## Functional form
Nonlinear in explanatory variables but still linear in parameters. Special cases:
- Polynomial regression $y_i = \beta_1 + \beta_2x_{i2} + \beta_3x_{i2}^2 + ... + \beta_r x_{i2}^{r-1} + \epsilon_i $
- Logarithms:
  - linear-log model: $y_i = \beta_1 + \beta_2\ln(x_i2) + \epsilon_i$
  $\rightarrow \Delta y = f(x_2+\Delta x_2)-f(x_2) \cong \beta_2 \frac{\Delta x_2}{x_2}$
  1% change in $x_2$ associated to $0.01\beta_2$ change in y
  - log-linear model: $\ln(y_i) = \beta_1 + \beta_ 2x_{i2} + \epsilon_i$
  $\rightarrow \frac{\Delta y}{y} \cong \beta_2 \Delta x_2$
  1 unit change in $x_2$ associated to $100\times\beta_2$% change in y
  - log-log model: $\ln(y_i) = \beta_1 + \beta_2\ln(x_i2) + \epsilon_i$
  $\rightarrow \frac{\Delta y}{y} \cong \beta_2\frac{\Delta x_2}{x_2}$
  1% change in $x_2$ associated to $\beta_2$% change in y
Note: cannot compare $R^2$ values of linear / linear-log and log-linear / log-log models because of different explanatory variables

**Interactions between Independent Variables**
For two independant variables
- if both binary: $y_i = \beta_1 + \beta_2D_{i1} + \beta_3D_{i2} + \beta_4(D_{i1}\times D_{i2}) + \epsilon_i$
- one binary one continuous: Three possibilities:
  - $y_i = \beta_1 + \beta_2x_{i2} + \beta_3D_{i} + \epsilon_i$
  - $y_i = \beta_1 + \beta_2x_{i2} + \beta_3D_{i} + \beta_4(x_{i2}\times D_{i}) + \epsilon_i$
  - $y_i = \beta_1 + \beta_2x_{i2} + \beta_3(x_{i2}\times D_{i}) + \epsilon_i$
- if both continuous: $y_i = \beta_1 + \beta_2x_{i2} + \beta_3x_{i3} + \beta_4(x_{i2}\times x_{i3}) + \epsilon_i$

## Specification analysis
Omitted Variable Bias: If the ommitted variable is significant and correlated with the regressor.
E.g. if in the structural model, $x_{i2}$ and $x_{i3}$ have causal effects on $y_i$. If we regress only on $x_{i2}$ and $\text{Corr}(x_{i2}, x_{i2}) \neq 0$ and $\beta_3 \neq 0$, $b_2$ will be inconsistent

Proxy variables: Variable correlated with an omitted factor we are not interested in. Can reduce omitted variable bias

Which variables should we use in our regression?
1. Identify the key coefficients of interest.
2. What are the most likely sources of important omitted variable bias?
3. Augment your base specification with the additional questionable variables and test the hypothesis that their coefficients are zero.
4. Full disclosure: present an accurate summary of your results.

Errors-in-Variables: independant variable inaccurately measured

Sample Selection bias: availability of the data is influenced by a selection process that is related to the value of the dependent variable $\rightarrow$ biased estimator

Simultaneous causality bias: causal link of interest from $x_2$ to $y$ and causal link from $y$ to $x_2$

Correlation of the Error Term across Observations $\rightarrow$ **(A2)** violated. 
Does not make the least squares estimator biased or inconsistent but standard errors are incorrect $\rightarrow$ confidence intervals incorrect

## The Generalized Regression Model and Heteroscedasticity
Previously **(A2)** i.i.d. data, **(A5)** conditional homoskedasticity.
Now: Relax these assumptions

Assumptions for the Generalized Regression Model:
- **(GLS1)** $E[\epsilon|X] = 0$
- **(GLS2)** $\text{Var}[\epsilon|X] = E[\epsilon\epsilon'|X] = \sigma^2\Omega(X) = \sigma^2\Omega$
- **(GLS3)** $x_i$ and $\epsilon_i$ satisfy suitable moment conditions
- **(GLS4)** $X$ has full column rank

**Finite sample properties**
$E[b|X] = \beta + (X'X)^{−1}X'E[\epsilon|X] = \beta$
$\text{Var}[b|X] = \sigma^2(X'X)^{−1}(X'\Sigma X)(X'X)^{−1}$
Assuming $\epsilon|X$ is normally distributed $\rightarrow b|X \sim \mathcal{N}(\beta, \sigma^2(X'X)^{−1}(X'\Sigma X)(X'X)^{−1}) $ 
Problem: b is still unbiased but inferences based on $s^2(X'X)^{−1}$ no longer valid and b is no longer the best linear unbiased estimator

Solutions: 
1. Generalized Least Squares $\rightarrow$ best linear unbiased estimator again
2. Robust standard errors $\rightarrow$ standard erros valid again

### Generalized Least Squares
Suppose we know $\Omega \rightarrow$ Cholesky decomposition $ \Omega^{-1}=P'P \rightarrow \underbrace{Py}_{y_\star} = \underbrace{PX}_{X_\star}\beta + \underbrace{P\epsilon}_{\epsilon_\star}$  
Now $\text{Var}[\epsilon_\star|X_\star] = \sigma^2I$ so we can run our normal regression on the transformed data
$\rightarrow$ Generalized Least Squares estimator of $\beta$
$$b_\star = (X_\star'X_\star)^{-1}X_\star'y_\star = (X'\Omega^{-1}X)X'\Omega^{-1}y$$
If there is no serial correlation, $\Omega$ is diagonal with entries $\omega_i$ $ \rightarrow$ Weighted Least Squares (WLS)
$$b_\star = (X_\star'X_\star)^{-1}X_\star'y_\star = 
\left( \sum_{i=1}^n \frac{1}{\omega_i}x_ix_i'\right)^{-1} \left( \sum_{i=1}^n \frac{1}{\omega_i}x_iy\right) $$

## Robust standard errors
Assume heteroskedasticity but no serial correlation, i.i.d. and **(A4a)**. 
Then $\sqrt{n}(b-\beta) \xrightarrow{d} \mathcal{N}(0, Q^{-1}\Sigma Q^{-1})$.
Under heteroskedasticity $\Sigma = E[\epsilon_i^2x_ix_i'] \rightarrow \text{AVar}[b] = \frac{1}{n}Q^{-1}\Sigma Q^{-1}$.
Estimate $\Sigma$:
$$\frac{1}{n-K}\sum_{i=1}^n e_i^2x_ix_i' \xrightarrow{P}\Sigma $$
Estimates for $Q$ and $\Sigma$ yield
$$\widehat{\text{AVar}}[b] = \frac{1}{n}\left(\frac{1}{n}\sum_{i=1}^n x_ix_i'\right)^{-1} \left(\frac{1}{n-K}\sum_{i=1}^n e_i^2x_ix_i'\right) \left(\frac{1}{n}\sum_{i=1}^n x_ix_i'\right)^{-1}$$
$\rightarrow$ White standard errors $se(b_k) = \sqrt{\widehat{\text{AVar}}[b]_{kk}}$

White's test for heteroskedasticity
$H_0: \sigma_i^2 = \sigma^2$
Regress $e_i$ on $z = \left(1, x_{i1},...,x_{ik}, x_{i1}^2, ..., x_{ik}^2\right)$
Obtain White’s test statistic $nR^2 \xrightarrow{d}\chi^2_{J-1}$ under $H_0$.
reject if $nR^2 > \chi_{J−1,1−\alpha}​$

with smaller sample sizes, the heteroskedasticity-robust statistics might not be well behaved