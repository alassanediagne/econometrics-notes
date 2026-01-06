- Define the Expected value
- $E[X] = \sum_{i}x_iP(X=X_i)$ or $E[X]=\int_{-\infty}^\infty x\cdot f_X(x)dx$

- Define variance
- $\text{Var}[x]=E[(X-\mu_X)^2] = E[X^2]-E[X]^2$

- State Jensen's inequality
- If g convex, then $g(E[X])\leq E[g(x)]$

- State the Expectation inequality
- $|E[X]| \leq E[|X|]$

- State the Cauchy-Schwarz inequality
- $|E[XY]|\leq \sqrt{E[X^2]E[Y^2]}$

- Define the rth raw moment and rth central moment
- $E[X^r]$, $E[(X-E[X])^r]$

- Define Skewness. What does it measure?
-  $\frac{E[(X-\mu_X)^3]}{\sigma_X^3}$. Measure of symmetry, if skewness = 0 symmetric

- Define Kurtosis. What does it measure? What is the curtosis of the standard normal
- $\frac{E[(X-\mu_X)^4]}{\sigma_X^4}$ Measure of tail mass. If kurtosis > 3 heavy-tailed

- Define the Conditional distribution
- $P(Y=y|X=x) = \frac{P(X=x, Y=y)}{P(X=x)}$ (discrete) $f_{Y|X=x}(y) = \frac{f_{X,Y}(x,y)}{f_X(x)}$ (continuous)

- State the Simple Law of iterated expectations
- $E[Y]=E[E[Y|X]]$

- State the Extended law of iterated expectations
- $E[Y|X]=E[E[Y|X,Z]|X]$

- State Variance decomposition law
- $\text{Var}[Y] = E[\text{Var}[Y|X]]+ \text{Var}[E[Y|X]]$

- When are two random variables independend?
- If $P(Y=y|X=x) = P(Y=y)$ or $f_{Y|X=x}(y)=f_Y(y)$

- Define Covariance and Correlation. What is the difference?
- Correlation is a standardized Covariance
    - Covariance: $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] = E[XY]-E[X]E[Y]$
    - Correlation: $\text{Corr}(X,Y)=\frac{\text{Var}(XY)}{\text{Var}(X)\text{Var}(Y)}$

- Give the definitions of Total Sum of Squares, Explained Sum of Squares and Sum of Squared Residuals
-
    - $TSS = \sum_{i=1}^n (y_i-\bar y)^2$
    - $ESS = \sum_{i=1}^n (\hat y_i-\bar y)^2$
    -  $SSR = \sum_{i=1}^n e_i^2$

- State the definition of the Coefficient of Determination $R^2$
- $$R^2 = \frac{ESS}{TSS} = 1-\frac{SSR}{TSS} = \widehat{corr}(y_i, x_{i2})^2$$ 

- Define the standard error of the regression
- $$s_e = \sqrt{ \frac{1}{n-2} \sum_{i=1}^n e_i} = \sqrt{\frac{SSR}{n-2}}$$

- State the Least Squares Assumptions for Prediction
- $(x_i, y_i)$ in sample data used to estimate regression coefficients, $(x_i^{OOS}, y_i^{OOS})$ out of sample data
$$E[y|x_2] = \beta_1 + \beta_2x_2,\quad \epsilon = y- E[y|x_2]$$
    - (A1) $(x_i^{OOS}, y_i^{OOS})$ are randomly drawn from the same population distribution as $(x_i, y_i)$
    - (A2) $(x_i, y_i)$ are i.i.d. draws from their joint distribution
    - (A3) $(x_i, \epsilon_i)$ have nonzero finite fourth moments

- State the Least Squares Aussumptions for Causal Inference
- structural model $y_i = \beta_1 + \beta_2 x_{i2} + \epsilon_i$
    - (A1) Conditional mean zero: $E[\epsilon_i|x_{i2}] = 0$
    - (A2) $(x_{i2}, y_i)$ are i.i.d. draws from their joint distribution
    - (A3) $(x_{i2}, \epsilon_i)$ have nonzero finite fourth moments
    - (A4) Homoskedasticity: $\text{Var}[\epsilon_i|x_{i2}] = \sigma_\epsilon^2$
    - (A5) Conditional normality: $\epsilon_i|x_{i2} \sim \mathcal{N}(0,\sigma_\epsilon^2)$

- Compare prediction and causal inference based on $R^2$ and biasedness
- Prediction vs causal inference:
    - Prediction
        - Goal: get Y right
        - High $R^2$ / low error = good
        - A biased model can still predict well

    - Causal inference:
        - Goal: estimate “what happens if we intervene”
        - Bias = fatal
        - A model with terrible $R^2$ can still give correct causal effects

