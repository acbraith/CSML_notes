---
# Computational Statistics and Machine Learning
---

## Statistics

---

### Classical Statistics

#### Introduction

#### Linear Models

- Inference
    - Parameter estimation
    - Variance estimation
    - Weighted Least Squares
    - Confidence Intervals
    - Hypothesis Tests
- Linear Regression
    - Basic Results
    - Interpretation
    - Model Adequacy
        - Residual Plots
        - $R^2$
        - Outliers
- Robust Regression
    - M-estimation
    - S-estimator
    - MM-estimator
- Variable Selection
    - t-tests and F-tests
    - Best subset selection
    - Comparing Models
    - Stepwise Methods
    - Lasso
- Analysis of Variance

#### Generalized Linear Models

- Exponential Family
    - Properties
- GLM Theory
    - Estimation
    - Confidence Intervals
    - Hypothesis Testing
- Binomial Data and logistic regression
    - Interpretation
    - Basic Results
    - Model Adequacy
    - Model Selection
    - Analysis of deviance

---

### Bayesian Statistics

#### Bayes Theorem

- Predictive Distribution
- Sequential Learning

#### Inference

- Functions of parameters
- Point & Interval estimates

#### Priors

- Conjugate
- Non-informative
    - Uniform
    - Jeffrey’s
- Hierarchical

#### Graphs

- DAG
- Moralise, conditional independence
- Factorisation theorem
- FUll conditional distributuion

#### Hierarchical Models

- Marginal prior
- Examples

#### Markov Chain Monte Carlo

- Gibbs sampling

---

### Stochastic Methods in Finance

---

## Machine Learning

---

### Supervised Learning

#### Linear Regression

- Least Squares
    - `D = (xᵢ,yᵢ) , i = 1,...,m , xᵢ∈Rⁿ`
        - `X = (x₁,x₂,...)ᵀ , y = (y₁,y₂,...)ᵀ`
    - Assume
        - `yᵢ ~ N(βᵀxᵢ, σ²I)`
    - Minimise square error
        - `E(β) = Σ(βᵀxᵢ-yᵢ)²`
        - `dE/dβ = 0 => XᵀXβ = Xᵀy`
            - Cannot be solved if `XᵀX` cannot be inverted (determinant = 0)
    - Robust Regression
        - L1, L2 regularisation (*weight decay*)
            - Add penalty term to error
                - `E(β) = Σ(βᵀxᵢ-yᵢ)² + λ|β|ₙ`
                - L1 Regularisation (*lasso*)
                    - Take L1 norm of `β`; encourages sparse solutions
                    - Harder to compute
                - L2 Regularisation (*ridge regression*)
                    - `dE/dβ = 0 => (XᵀX+λI)β = Xᵀy`
                        - `(XᵀX+λI)` always invertible

#### Generalised Linear Models

- Components
- Exponential Family
- Logistic Regression

#### Kernel Methods

- Kernel Ridge Regression
    - Transform input into higher dimensional space using basis function
        - `φ:Rⁿ→Rᴺ`
    - Allow non-linear regression
    - Kernel Trick: No need to do expensive high-dimensional calculations
        - `ŷ = βᵀx`
            - *Primal Form*
        - `β = 1/λ Σ(βᵀxᵢ-yᵢ)`
            - Ridge regression solution (`dE/dβ = 0`)
        - `=> β = Σαᵢxᵢ` (`αᵢ = 1/λ (βᵀxᵢ-yᵢ)`)
        - `=> ŷ = Σαᵢxᵢx`
            - *Dual Form*: Good when `m<<n` (ie high dimensional data)
        - `=> ŷ = Σαᵢ<φ(xᵢ)φ(x)> = ΣαᵢK(xᵢ,x)`
            - No need for basis function; only need Kernel Function
                - `K:RⁿxRⁿ→R , K(x,t) = <φ(x),φ(t)>`
                    - `K` is positive semi-definite (PSD => symmetric)
                        - `K` is PSD => `K` corresponds to *at least* one basis function `φ`
    - Example Kernels:
        - Polynomial
            - `K(x,t) = (1 + xᵀt)ᵈ`
        - Gaussian
            - `K(x,t) = exp(-β|x-t|²)`
            - Infinite dimensional corresponding feature map!
- Support Vector Machine
    - Classification
        - Find separating hyperplane
            - `Hᵦₖ = [ x∈Rᵈ.βᵀx+k=0 ]`
                - classify `ŷ = 1` if `βᵀx+k > 0`, `ŷ = -1` otherwise
        - We want to find maximially separating hyperplane
            - Find `β,k` to maximise the minimum distance of any `xᵢ` from `Hᵦₖ`
            - Requires minimising `βᵀβ` subject to `∀i.yᵢ(βᵀx+k)≥1`
        - Linearly non-separable case
            - Add *slack variables* `εᵢ` to allow some points to have margin < 1
            - Now minimise `βᵀβ+CΣεᵢ` subject to `∀i.yᵢ(βᵀx+k)≥1-εᵢ,εᵢ≥0`
                - `C` controls trade-off between `βᵀβ` and training error
        - Kernels
            - Can use feature map `φ` and *Kernel trick* to perform in higher dimensional space
    - Regression
        - New loss function: `L(y,ŷ) = max(|y-ŷ|-ε,0)`
            - Errors below `ε` ignored, leading to sparse solutions!
- Gaussian Process
    - **TODO this section can be improved; look at dissertation notes**
    - Outputs corrupted by additive noise
        - `y|β,x ~ N(<β,φ(x)>, σ²)`
    - MAP Inference
        ```
           dP(β|D) = dP(β) P(y|D,β)
             dP(β) α exp(-|β|²/2)
          P(y|D,β) α exp((-Σ(<β,xᵢ>-yᵢ)²)/(2σ²))
        => dP(β|D) α exp(-|β|²/2) exp((-Σ(<β,xᵢ>-yᵢ)²)/(2σ²))
           dP(β|D) = 0 => βₘₐₚ = Xᵀ(K+σ²I)⁻¹y
        ```
        - `K` is the Kernel Matrix; `Kₓₜ=K(x,t)`
        - `dP(β)` is the prior on `β`, `P(y|D,β)` is the likelihood, or noise model
        - `P(β|D)` is the posterior. `βₘₐₚ` is the MAP estimator for `β`
    - Inference:
        ```
        P(β|D) α exp((-Σ(<β,xᵢ>-yᵢ)²)/(2σ²) - |β|²/2)
               α exp(-1/2 (β-βₘₐₚ)ᵀ Σ⁻¹ (β-βₘₐₚ))
        => β|D ～ N(βₘₐₚ, Σ) (Σ⁻¹ = 1/σ² (XᵀX+σ²I))
        ```



#### Decision Trees

- Classification and Regression Trees (*CART*)
    - Divide space into hyperrectangles by recursive binary partitions
    - Regression
        - `f(x) = mean(yᵢ|xᵢ∈Rₓ)`
        - Aim to find regions `R` to minimise squared error of `f`
            - Computationally intractable to solve
        - Greedy approximation
            - Each iteration, find best binary split to minimise squared data
                - `O(dm)` for `d` dimensions and `m` data points
            - Recurse until each node contains max of `n` data points
        - Regularisation
            - Successively collapse nodes giving smallest per-node increase in training error
            - From sequence of trees `T₁, T₂, T₃...` select tree minimising `<Train Error>(T) + λ|T|`
    - Classification
        - 

#### Ensemble Methods

- Bagging
- Boosting

#### Neural Networks

- Deep Network Layers
- Convolutional Networks
- Recurrent Networks
- Attention and Memory
- Optimisation

#### Feature Selection

- t-tests, F-tests
- Akaike Information Criteria, Cross-Validation
- Stepwise methods
    - Backward & Forward
- Select from model
    - L1 Based
    - Tree Based

---

### Unsupervised Learning

#### Dimensionality Reduction

- PCA, PPCA, FA, ICA
- Fishers LDA
- t-SNE

#### Clustering

- K-means
- Mixture models
- Spectral clustering

---

### Reinforcement Learning

#### Markov Decision Process

- Agent, Environment, State
    - Agent conponents: policy, value function, model
- Bellman equations

#### Dynamic Programming approaches

- Policy evaluation
- Policy iteration
- Value iteration

#### Model-Free

- Monte-Carlo & Temporal-Difference policy evaluation
- Sarsa
- Q-Learning
- Policy Gradient
    - REINFORCE
- Actor-Critic

#### Model-Based

- Dyna
- MCTS

#### Exploration vs Exploitation

- ε-greedy
- Upper Confidence Bounds
