
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

- Causal effect $E[Y|X=1] - E[Y|X=0]$ 
- Total Sum of Squares $TSS = \sum_{i=1}^n (y_i-\bar y)$
- Explained Sum of Squares $ESS = \sum_{i=1}^n (\hat y_i-\bar y)$
- Sum of Squared Residuals $SSR = \sum_{i=1}^n e_i$
- Coefficient of Determination 
  $$R^2 = \frac{ESS}{TSS} = 1-\frac{SSR}{TSS} = \widehat{corr}(y_i, x_{i2})^2$$ 
  -> high $R^2$ is good
- Standard error of the regression (Stata: Root MSE) 
  $$s_e = \sqrt{ \frac{1}{n-2} \sum_{i=1}^n e_i} = \sqrt{\frac{SSR}{n-2}}$$

---

**OLS Assumptions**

**Least Squares Assumptions for Prediction:**
$(x_i, y_i)$ in sample data used to estimate regression coefficients, $(x_i^{OOS}, y_i^{OOS})$ out of sample data
$$E[y|x_2] = \beta_1 + \beta_2x_2,\quad \epsilon = y- E[y|x_2]$$
- (A1) $(x_i^{OOS}, y_i^{OOS})$ are randomly drawn from the same population distribution as $(x_i, y_i)$
- (A2) $(x_i, y_i)$ are i.i.d. draws from their joint distribution
- (A3) $(x_i, \epsilon_i)$ have nonzero finite fourth moments

**Least Squares Aussumptions for Causal Inference**
structural model $y_i = \beta_1 + \beta_2 x_{i2} + \epsilon_i$
- (A1) Conditional mean zero: $E[\epsilon_i|x_{i2}] = 0$
- (A2) $(x_{i2}, y_i)$ are i.i.d. draws from their joint distribution
- (A3) $(x_{i2}, \epsilon_i)$ have nonzero finite fourth moments
- (A4) Homoskedasticity: $\text{Var}[\epsilon_i|x_{i2}] = \sigma_\epsilon^2$
- (A5) Conditional normality: $\epsilon_i|x_{i2} \sim \mathcal{N}(0,\sigma_\epsilon^2)$

---

- Treatment effect $te_i = Y_i(1)-Y_i(0)$ (but: we usually cannot observe both)
- Average Treatment Effect (ATE/ACE): $\tau_{ATE}=E[te_i] = E[Y_i(1)-Y_i(0)] = E[Y_i(1)]-E[Y_i(0)]$
- Average Treatment Effect on the Treated (ATET): $\tau_{ATET} = E[Y_i(1) | X_i=1] = E[Y_i(0) | X_i=1]$
- Average Treatment Effect on the Non-Treated (ATENT): $\tau_{ATENT} = E[Y_i(1) | X_i=0] = E[Y_i(0) | X_i=0]$
- ATE = $\pi\tau_{ATET} + (1 − \pi) \tau_{ATENT}$
- 
  $$ E[Y_i|X_i = 1] − E[Y_i|X_i = 0]\\ = \tau_{ATE} + \underbrace{E[Y_i(0)|X_i=1]-E[Y_i(0)|X_i=0]}_{\text{selection bias}}\\ + \underbrace{(1 − \pi) (\tau_{ATET}-\tau_{ATENT})}_{\text{heterogeneous treatment effect bias}} $$ 
- -> Randomized controlled experiment (RCT): random entities and random assignments
- Under random assignment selection bias and heterogeneous treatment effect bias = 0 -> $E[Y_i|X_i = 1] − E[Y_i|X_i = 0] =\tau_{ATE}$
- relation to linear reg.: $\beta_1 = E[Y_i(0)], \beta_2 = \tau, \epsilon_i = \epsilon_i(0)$
- multiple linear regression: $y=X\beta+\epsilon$

**Least Squares Aussumptions for Causal Inference (multidimensional)**
structural model $y_i = \beta_1 + \beta_2 x_{i2} + \epsilon_i$
- (A1) Conditional mean zero: $E[\epsilon_i|x_{ik}] = 0, \quad k=1,...,K$
- (A2) $(x_{i2}, y_i)$ are i.i.d. draws from their joint distribution
- (A3) $x_{i}, \epsilon_i$ have nonzero finite fourth moments
- (A4) Full rank: $\text{rank}(X) = K$
- (A5) Conditional homoskedasticity: $\text{Var}[\epsilon_i|x_i] = E[\epsilon_i^2|x_i]= \sigma^2$
- (A6) Conditional normality: $\epsilon_i|x_{i2} \sim \mathcal{N}(0,\sigma_\epsilon^2)$

- -> Least squares: 
  $$\min_{b_0} e_0'e_0 = (y-Xb_0)'(y-Xb_0)$$
- b satisifies normal equations $(X'X)b = X'y$ -> $b = (X'X)^{-1}X'y$
- Adjusted $R^2$: $R^2$ increases with $K$ -> $\bar R^2 = 1-\frac{n-1}{n-K}(1-R^2)$
- 