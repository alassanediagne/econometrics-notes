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

- 