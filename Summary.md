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

### Maths Basics

- **TODO See if anything to add from Matrix Cookbook and Unsupervised Notes**
- Linear Algebra
    - Matrix Determinant, Trace, Inverse, Rank
    ```
    A = |a b| => det(A) = ad-bc
        |c d|     
      det(Aᵀ) = det(A)
      det(AB) = det(A) det(B)
     det(A⁻¹) = 1/det(A)
        tr(A) = Σᵢaᵢᵢ
    tr(log A) = log det(A)
         A⁻¹A = I
       (AB)⁻¹ = B⁻¹A⁻¹
       (ABC)ᵀ = CᵀBᵀAᵀ
    ```
    - Orthogonality
        - `AᵀA = I = AAᵀ => Aᵀ = A⁻¹ <=> A` is orthogonal
    - Eigenvectors / Eigenvalues
        ```
        Ae = λe
            e: eigenvectors
            λ: eigenvalues
        (A-λI)e = 0 => λ is eigenvalue if det(A-λI) = 0
        A = Σᵢeᵢeᵢᵀ
            (eigenvectors are orthogonal to each other; ∀i≠j.eᵢᵀeⱼ = 0)
         tr(A) = Σᵢaᵢᵢ = Σᵢλᵢ
        det(A) = Πᵢλᵢ
        ```
    - Singular Value Decomposition
        ```
        X = USVᵀ
            X: nxp
            U: nxn, UᵀU=Iₙ
            V: pxp, VᵀV=Iₚ
            S: nxp, S is diagonal, all +ve and ordered (largest top left)
        ```
    - Positive (Semi) Definite
        ```
        PSD if ∀z.zᵀAz ≥ 0 (PD if > 0)
        ```
        - eg `I` is PSD
        - Negative (Semi) Definite analogous
- Calculus
    - Differentiation
        ```
        df/dx = lim_{δ->0} [f(x+δ)-f(x)] / δ = f'(x)
        ```
    - Taylor Series
        `f(x) = f(0) + x df/dx + x²/2! d²f/dx² + ...`
    - Chain Rule
        ```
        df(g(x))   df . dg
          dx     = dg   dx
        ```
    - Matrix Calculus
        - Jacobian, `∇f(x)`
            - (Column) vector of first derivatives
        - Hessian, `H_f(x)`
            - Matrix of second derivatives
            - `Hᵢⱼ=d²f/dxᵢxⱼ`
        ```
           d/dA tr(AB) = Bᵀ
          d log det(A) = d tr(logA) = tr(A⁻¹ dA)
        d/dA log det A = Aᵀ
                  dA⁻¹ = -AᵀdAA⁻¹
        ```
- Convex Analysis
- Numerical Issues
- Distributions
    - Multi-variate Gaussian


### Supervised Learning

#### Regression

- Least Squares
    - `D = (xᵢ,yᵢ) , i = 1,...,m , xᵢ∈Rⁿ`
        - `X = (x₁,x₂,...)ᵀ , y = (y₁,y₂,...)ᵀ`
            - ie for `X`, each row is an `xᵢ`
    - Assume
        - `yᵢ ~ N(βᵀxᵢ, σ²I)`
    - Minimise square error
        - `E(β) = Σ(βᵀxᵢ-yᵢ)²`
        - `dE/dβ = 0 => XᵀXβ = Xᵀy`
            - Cannot be solved if `XᵀX` cannot be inverted (determinant = 0)
    - Robust Regression
        - L1, L2 regularisation (*weight decay*)
            - First scale input dimensions to unit variance
            - Add penalty term to error
                - `E(β) = Σ(βᵀxᵢ-yᵢ)² + λ|β|ₙ`
                - L1 Regularisation (*lasso*)
                    - Take L1 norm of `β`; encourages sparse solutions
                    - Non-differentiable; need special optimisation routines
                - L2 Regularisation (*ridge regression*)
                    - `dE/dβ = 0 => (XᵀX+λI)β = Xᵀy`
                        - `(XᵀX+λI)` always invertible
- Kernel Ridge Regression, Gaussian Process
    - (GPs can also be used for classification)

#### Classification

- Generative Models
    - Nearest Neighbours
        - Classify `x` using `k` nearest points in training set
        - Distances:
            - Squared Euclidean: `d(x,t) = (x-t)ᵀ(x-t)`
                - Loses information if length scales of components of `x` vary
            - Mahalanobis: `d(x,t) = (x-t)ᵀΣ⁻¹(x-t)` (`Σ` is covariance of all inputs)
                - Rescales components of `x`
        - Gives smooth boundaries, but slow and large space requirements
        - Classifies distant test points very confidently
    - Naive Bayes
        - `P(c|x) = P(x|c) P(c) / P(x)` (`P(x) = ΣₖP(x|k)P(k)`)
        - Fast to train; just need to count for discrete attributes
        - Overly confident with low counts; add pseudo counts for classes to help reduce this issue
- Discriminative Models
    - Logistic Regression
        - **TODO From Applied ML Notes**
    - SVM, Decision Trees, Ensemble methods
        - (all can also be used for regression)

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
            - From sequence of trees `T₁, T₂, T₃...` select tree minimising `<Train Error>(Tᵢ) + λ|Tᵢ|`
    - Classification
        - Growing Tree
            - Minimise one of
                - Gini Index `= Σₖρₙₖ(1-ρₙₖ)`
                - Cross Entropy `= Σₖρₙₖlog(1/ρₙₖ)`
                    - `ρₙₖ = 1/m ΣₓI(yᵢ=k)` (empirical class `k` probability for region `n`)
        - Pruning Tree
            - Prune smallest misclassification error `= 1 - argmaxₖρₙₖ`

#### Ensemble Methods

- Bagging
    - Reduce **variance** by averaging over ensemble
        - Use high variance but low bias estimators; eg decision trees
    - Train T variations, each on `M` examples (sampled *with* repleacement => ~63% non-repeats)
        - Conventially, `M` is size of training set
    - Random Forest
        - Decision Trees high variance; use bagging to reduce this
        - Can train each tree with only a subset of `k` features; further decorrelates trees in ensemble
            - Usually `k = √d` or `k = log(d)`
- Boosting
    - Reduce **bias** by focusing on hard examples
        - Use lower variance but high bias estimators; eg decision stumps
    - Adaboost
        - Initialise `D₁(1),...,D₁(m) = 1/m` (training point weights)
        - for `t = 1,...,T`
            ```
             Fit hₜ : Rᵈ->(-1,1) (using Dₜ)
                 αₜ = 1/2 log((1-εₜ)/εₜ) , εₜ=weighted training error
            Dₜ₊₁(i) α Dₜ(i) exp(-αₜyᵢhₜ(xᵢ)) (ΣᵢDₜ₊₁(i) = 1)
            ```
        - Return `H(x) = sign(Σₜαₜhₜ(x))`
            - Linear combination of weak learners, weighted by training errors
        - Adaboost greedily solves
            - `min[ΣᵢL(yᵢ,Σₜαₜhₜ(xᵢ))]`, where `L(y,ŷ) = exp(-yŷ)`
                - Exponential loss punishes negative margins
                - Minimising exponential loss for `hₜ` with fixed `αₜ` equivalent to minimising misclassification error

#### Neural Networks

- **TODO Use Advanced ML Notes**
- Deep Network Layers
- Convolutional Networks
- Recurrent Networks
- Attention and Memory
- Optimisation
    - First Order Methods
        - Gradient Descent
            - `βₖ₊₁ = βₖ - α∇f(βₖ)`
                - Easy to get stuck in local optima
                - Too big `α` -> Could diverge
        - Momentum
            - `gₖ₊₁ = μₖgₖ - α∇f(βₖ)`
            - `βₖ₊₁ = βₖ + gₖ₊₁`
                - Good for noisy gradients, helps with crossing flat regions of gradient (eg saddle points)
        - Adam
            - Use running average of gradient and second moment of gradient, with some clever normalisation
        - Line Search
            - Pick direction (eg steepest direction) and find optimum in given direction
    - Second Order Methods
        - Newton's Method
            - Taylor expand `f(β+Δ)` to second order
            - `βₖ₊₁ = βₖ - α H⁻¹∇f`
                - Converge in 1 step for quadratic function, `α=1`
                - Invariant under coordinate transform
                - Calculating `H⁻¹∇f` and storing `H` expensive
                - Not guaranteed to produce downhill step
                    - Line search in `H⁻¹∇f` direction instead
    - Large Scale Learning
        - Online Stochastic Gradient Descent
            - Train on mini-batches
            - Very parallelisable and robust to synchronisation issues
            - More robust than training on whole training set each iteration; can help reduce overfitting

#### Feature Selection

- **TODO from Statistics Notes**
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
