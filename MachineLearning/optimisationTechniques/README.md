Gradient descent and second-order derivative methods are optimization techniques widely used in machine learning and other areas. Let's compare them:

**1. Vanilla Gradient Descent:**
- **Basic Principle:** Adjust the weights (or parameters) in the opposite direction of the gradient of the loss function with respect to the weights.
- **Update Rule:** w=w−α∇J(w) where α is the learning rate and ∇J(w) is the gradient.
- **Pros:**
  - Simplicity: The method is straightforward and easy to implement.
  - Scalability: Works well for large-scale datasets and high-dimensional problems.
- **Cons:**
  - Requires choosing a proper learning rate: Too large and it might diverge; too small and it might converge slowly.
  - Might get stuck in saddle points.
  - Convergence rate is not as fast for ill-conditioned functions.

**2. Second-Order Derivative Methods (e.g., Newton's method):**
- **Basic Principle:** Use both the first and second derivatives to update the weights. 
- **Update Rule:** w=w−[∇^(2)J(w)] ^(−1) ∇J(w) where ∇^(2)J(w) is the Hessian (second-order derivative matrix).
- **Pros:**
  - Faster convergence for certain functions: Especially for quadratic or near-quadratic functions.
  - Doesn't require setting a learning rate in its pure form (though it's often combined with line search or trust region methods).
  - Can deal better with certain saddle points.
- **Cons:**
  - Computationally intensive: The Hessian needs to be computed (which can be large and expensive for high-dimensional problems) and inverted.
  - Not scalable for very large datasets or high-dimensional problems due to memory and computational constraints.
  - Could converge to a saddle point, but this is less of an issue than with first-order methods.
  - Requires a good initial value to ensure convergence to a good solution; bad initializations might lead to poor local minima or other issues.

**Regarding good initial values:**
If you have a good initial guess, second-order methods are more likely to converge quickly and to a better solution than vanilla gradient descent. This is because the inclusion of curvature information (from the Hessian) can help direct the optimization more effectively. However, the "goodness" of the initial value also matters for first-order methods.

**In practice:**
While second-order methods can offer superior convergence properties, their computational requirements often make them impractical for very large-scale problems or deep learning models. In these cases, variations of first-order methods, like momentum, RMSprop, Adam, etc., are often used because they offer a good trade-off between convergence speed and computational efficiency.

In summary, the right optimization technique depends on the problem size, the nature of the function, computational constraints, and sometimes even empirical trial and error.

In the code attached, we will try to validate the above hypothesis or theoretical point we obtained from the literature.
We will try to find the optimal value for the minimum value of Rosenbrock, Rastrigin, Himmelblau and Eggholder functions.
