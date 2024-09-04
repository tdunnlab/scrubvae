**SC-VAE-MALS** Consider a linear decoder, $f_{\psi}(\mathbf{z})=\psi\mathbf{z}=\mathbf{\hat{v}}$, where $\psi=\psi^{0}(\psi^{1})^{-1}$, which can be evaluated by $L(\mathbf{z},\mathbf{v}; \psi)=||\mathbf{v}-f_{\psi_a}(\mathbf{z})||^2_2$.

$\phi,\theta\leftarrow$ Init network parameters

$\psi_a,\psi_b\in\mathbb{R}^{N\times D}\leftarrow$ Init parameters of two linear decoders

Initialize forgetting factors with fixed offset, $\epsilon$

$\lambda_a\leftarrow\alpha\in(0,1-\epsilon)$

$\lambda_b\leftarrow\lambda_a+\epsilon$

**repeat**

Draw batch with $K$ samples: $(\mathbf{x_k},\mathbf{v_k}\in\mathbb{R}^{N\times K})$

$\mathbf{z_k}\sim q_\phi(\cdot\mid\mathbf{x_k})\in\mathbb{R}^{D\times K}$

Calculate loss by decoder and average for scrubbing loss

$L_\text{scrub}=-\frac{1}{2}[L(\mathbf{z_k},\mathbf{v_k};\psi_a)+L(\mathbf{z_k},\mathbf{v_k};\psi_b)]$

Forgetting factors step by $\Delta$ in the direction of the better decoder

**if** $L(\mathbf{z_k},\mathbf{v_k};\psi_a)>L(\mathbf{z_k},\mathbf{v_k};\psi_b)$ **then** $\lambda_a=max(\lambda_a-\Delta,0),\lambda_b=\lambda_a+\epsilon$

**else** $\lambda_b=min(\lambda_b+\Delta,1),\lambda_a=\lambda_b-\epsilon$

Update $\psi_a$ and $\psi_b$ based on the normal equations for ordinary least squares regression

**for** i = [a,b]

$\psi_i=[\mathbf{v_k}\mathbf{z_k}^T+\lambda_i\psi_i^{0}]\left([\mathbf{z_k}\mathbf{z_k}^T+\lambda_i\psi_i^{1}]\right)^{-1}$

**end for**

Update network parameters

$\phi \leftarrow \phi + \nabla[L_\text{scrub} + L_\text{ELBO} + L_\text{Recon}]$

$\theta \leftarrow \theta + \nabla[ L_\text{Recon}]$

**until** convergence

**Algorithm 2: SC-VAE-QD** Consider the class-conditional Bayes classifier, $f_{\psi}(\mathbf{z})=p(v=c|\mathbf{z})$, with likelihood, $p(z|v=c)=N(\mathbf{z}|\mu^{c},\Sigma^{c})$. For multi-class problems, we maintain a *one vs rest* estimator per class, $\psi = \{ \mu^{c}, \Sigma^{c},\mu^{c'}, \Sigma^{c'}\ \forall\ c\in C\}$. This estimator can be evaluated based on the Gaussian log-likelihood, $L(\mathbf{z},v;\psi^{c})=\ell(\mu_{a}^{c},\Sigma_{a}^{c}|\mathbf{z},v=c)+\ell(\mu_{a}^{c'},\Sigma_{a}^{c'}|\mathbf{z},v\neq c)$.

$\phi,\theta\leftarrow$ Init network parameters

$\psi_a,\psi_b\leftarrow$ Init parameters of two classifers

$\lambda_a\leftarrow\alpha\overrightarrow{\mathbf{1}}_C,\ \alpha\in(0,1-\epsilon)$

$\lambda_b\leftarrow\lambda_a+\epsilon\overrightarrow{\mathbf{1}}_C$

**repeat**

Draw minibatch of $(\mathbf{x_k},\mathbf{v_k}, \mathbf{z_k})$ as in Alg 1

$L_\text{scrub}=0$

**for** $c\in C$

Evaluate the Gaussian log-likelihood ratio for each quadratic classifier and average for the scrubbing loss

$L_\text{scrub}=L_\text{scrub}+\frac{1}{2K}\sum_K [L(\mathbf{z}_k,v_k;\psi^{c}_a)+L(\mathbf{z}_k,v_k;\psi^{c}_b)]$

Update $\lambda_{a}^{c}$ and $\lambda_{b}^{c}$ based on $\frac{1}{K}\sum_K L(\mathbf{z}_k,v_k;\psi^{c}_a)$ and $\frac{1}{K}\sum_K L(\mathbf{z}_k,v_k;\psi^{c}_b)$ as in Alg 1

Update class means and covariances for both estimators

**for** $i=[a,b]$

$\mu_i^{c}=\mathbb{E}_{\mathbf{v_k}=c} [\mathbf{z_k}]+\lambda_i^{c}\mu^{c}$

$\Sigma_i^{c}=\text{Cov}_{\mathbf{v_k}=c}[\mathbf{z_k},\mathbf{z_k}]$

$\mu_i^{c'}=\mathbb{E}_{\mathbf{v_k}\neq c}[\mathbf{z_k}]+\lambda_i^{c'}\mu^{c'}$

$\Sigma_i^{c'}=\text{Cov}_{\mathbf{v_k}\neq c}[\mathbf{z_k},\mathbf{z_k}]$

**end for**

**end for**

Update network parameters as in Alg 1

**until** convergence

### Algorithm 3: Automatically tuning forgetting factor ($\lambda$)
Given a discriminator function, $f_{\psi}(\mathbf{z})$, and a minimization objective, $L(z, v; \psi)$, we can automate tuning of the forgetting factor of the moving average algorithms by simultaneously estimating two discriminators from the same function family, $f_{\psi_a}(\mathbf{z})$ and $f_{\psi_b}(\mathbf{z})$.

$\psi_a, \psi_b \leftarrow$ Initialize parameters of two discriminators

Initialize forgetting factors with fixed offset, $\epsilon$

$\lambda_a \leftarrow \alpha \in (0, 1-\epsilon)$

$\lambda_b \leftarrow \lambda_a + \epsilon$

**repeat**

Draw minibatch with $K$ samples: $(\mathbf{x_k}, \mathbf{v_k})$

$\mathbf{z_k} \sim q_\phi(\cdot \mid \mathbf{x_k}) \in \mathbb{R}^{D\times K}$

Average the losses of the two discriminators to obtain the scrubbing loss

$L_\text{scrub} = \frac{1}{2}[L(\mathbf{z_k}, \mathbf{v_k}; \psi_a) + L(\mathbf{z_k}, \mathbf{v_k}; \psi_b)]$

Forgetting factors step by $\Delta$ in the direction of the better decoder between $f_{\psi_a}$ and $f_{\psi_b}$

**if** $L(\mathbf{z_k}, \mathbf{v_k}; \psi_a) > L(\mathbf{z_k}, \mathbf{v_k}; \psi_b)$ 

$\lambda_a = max(\lambda_a - \Delta, 0), \lambda_b = \lambda_a + \epsilon$

**else** 

$\lambda_b = min(\lambda_b + \Delta, 1), \lambda_a = \lambda_b - \epsilon$

**end if**

Continue with updating $\psi_a$, $\psi_b$, $\phi$, and $\theta$ as described in Algorithms 1 and 2.

**until** convergence