**Reinforcement Learning can be used to solve problems with arbitrarily large state spaces. Examples:**
* Computer based Go game: $10^{170}$ states
* Helicopter operation: Continuous state spaces

Using tabular methods to store the value functions is not practical for these problems. So, far we have represented value function by a table/matrix where, every state $x_k$ has an entry $V(x_k)$ or every state-action pair $(x_k, u)$ has an entry $q(x_k, u_k)$.

**Problem with large MDPs**
* There is too many states and/or action to store in memory.
* It is too slow to learn the values of each state individually. 

**Solution of Large MDPs** \
Our goal is to find a good approximate solution using limited computational resources.
* Estimate value function with function approximation (function approximation takes examples from a desired function (e.g., a value function) and attempts to generalize from them to construct an approximation of the entire function). \
$$\hat{V}(x, \bold{w}) \simeq V_{\pi}(x) \quad or \quad \hat{q}(x, u, \bold{w}) \simeq q_{\pi}(x, u)$$
* Generalization from visited states to unseen states.
* Update parameter $w$ using MC and TD learning.

**Which function approximator?**
* Linear Combination
* Multi-Layer Neural Network
* Decision Tree
* Nearest Neightbour

**Problems with function approximation**
* Reinforcement learning with function approximation involves a number of new issues that do not normally arise in conventional supervised learning, such as nonstationarity, bootstrapping, and delayed targets.

**Predictive Objective** \
Up to now, we have never specified an explicit objective for prediction/estimation. In the tabular case a continuous measure of prediction quality was not necessary because the learned value function could come to equal the true value function exactly. Moreover, the learned values at each state were decoupled—an update at one state affected no other states. 

But, with function approximation, an update at one state affects many others, and it is not possible to get the values of all states exactly correct and so, making one state’s estimate more accurate invariably means making
other's less accurate. 

Mean Squared Value Error: It is the square of the difference between approximate value $\hat{V}(x, \bold{w})$ and the true value $V_{\pi}(x)$, we have obtain a natural objective function, the mean square value error, 
$ \sum_{x \in \chi}[V_{\pi}(x) - \hat{V}(x, \bold{w})]^2$

**Goal**: Find the parameter vector $\bold{w}$ to minimize the mean square value error:
$$J(\bold{w}) = \mathbb{E}_{\pi}[V_{\pi}(x) - \hat{V}(x, \bold{w})]^2$$

To, adjust the parameter $w$ in the direction of -ve gradient of $J(\bold{w})$,
$$\nabla{w} \propto \nabla{J(\bold{w})}$$
$$\nabla{w} = -\frac{\alpha}{2} \nabla{J(\bold{w})}$$
$$\nabla{w} = -\frac{\alpha}{2} \frac{\partial {J(\bold{w})}}{\partial \bold{w}}$$
$$\nabla{w} = -\frac{\alpha}{2} \frac{\partial}{\partial \bold{w}} \mathbb{E}_{\pi}[V_{\pi}(x) - \hat{V}(x, \bold{w})]^2$$
$$w \leftarrow w+ \alpha \mathbb{E}_{\pi}[V_{\pi}(x_k) - \hat{V}(x_k, \bold{w})]*\nabla{\hat{V}(x_k, \bold{w})}$$

for, monte-carlo based method,
$$w \leftarrow w+ \alpha \mathbb{E}_{\pi}[G_k - \hat{V}(x_k, \bold{w})]*\nabla{\hat{V}(x_k, \bold{w})}$$

for, temporal-difference based method,
$$w \leftarrow w+ \alpha \mathbb{E}_{\pi}[r_{k+1} + \gamma*\hat{V}(x_{k+1},\bold{w}) - \hat{V}(x_k, \bold{w})]*\nabla{\hat{V}(x_k, \bold{w})}$$