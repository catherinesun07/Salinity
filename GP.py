import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.linalg import cho_solve
import pandas as pd

# ==========================================
# 1. THE WORLD (Black Box Test Function)
# ==========================================
class TestFunction:
    """
    This represents the 'Real World'. 
    The GP does NOT know about this class or its internals.
    """
    def __init__(self, a, c, x_min, x_max, y_min, y_max, noise_std):
        self.noise_std = noise_std
    
    def analytical_function(self, x, noise_std=0.0):
        # The ground truth function (hidden from GP)
        # y = 50 * x / (x + 100)
        return 50.0 * (x / (x + 100.0)) + np.random.normal(0, noise_std, size=x.shape)

    def sample(self, df):
        x = df['x'].values.reshape(-1, 1)
        y = self.analytical_function(x, self.noise_std)
        return pd.DataFrame({'x': x.flatten(), 'salinity': y.flatten()})

# ==========================================
# 2. THE BRAIN (Robust Gaussian Process)
# ==========================================
class GaussianProcess:
    def __init__(self, kernel='matern32', n_restarts=10, optimizer='L-BFGS-B', jitter=1e-6):
        self.kernel = kernel
        self.n_restarts = n_restarts
        self.optimizer = optimizer
        self.jitter = jitter
        self.theta = None
        
        # Normalization parameters
        self.X_mean, self.X_std = 0.0, 1.0
        self.y_mean, self.y_std = 0.0, 1.0

    def _matern32_kernel(self, X1, X2, length_scale, signal_var):
        dist = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2))
        sqrt3_r_l = np.sqrt(3) * dist / length_scale
        return signal_var * (1 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
    
    def _matern52_kernel(self, X1, X2, length_scale, signal_var):
        dist = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2))
        sqrt5_r_l = np.sqrt(5) * dist / length_scale
        return signal_var * (1 + sqrt5_r_l + (5/3) * (dist**2) / (length_scale**2)) * np.exp(-sqrt5_r_l)

    def Kernel(self, X1, X2, length_scale, signal_var):
        if self.kernel == 'matern32':
            return self._matern32_kernel(X1, X2, length_scale, signal_var)
        else:
            return self._matern52_kernel(X1, X2, length_scale, signal_var)

    def Neglikelihood(self, theta_log10):
        sigma2_noise = 10**theta_log10[0]
        signal_var = 10**theta_log10[1]
        length_scale = 10**theta_log10[2]

        n = self.X_norm.shape[0]
        K = self.Kernel(self.X_norm, self.X_norm, length_scale, signal_var) 
        K += np.eye(n) * (sigma2_noise + self.jitter)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e9 

        alpha = cho_solve((L, True), self.y_norm)
        log_likelihood = -0.5 * (self.y_norm.T @ alpha) - np.sum(np.log(np.diag(L))) - (n / 2) * np.log(2 * np.pi)
        
        self.current_K = K
        self.current_L = L
        return -log_likelihood.flatten()

    def fit(self, X, y):
        self.X_mean, self.X_std = np.mean(X), np.std(X)
        if self.X_std < 1e-8: self.X_std = 1.0
        self.X_norm = (X - self.X_mean) / self.X_std

        self.y_mean, self.y_std = np.mean(y), np.std(y)
        if self.y_std < 1e-8: self.y_std = 1.0
        self.y_norm = (y - self.y_mean) / self.y_std

        lb = np.log10([1e-6, 0.05, 0.01]) 
        ub = np.log10([0.5,  5.0,  10.0]) 
        bnds = Bounds(lb, ub)

        best_res = None
        for i in range(self.n_restarts):
            theta0 = np.random.uniform(lb, ub)
            res = minimize(self.Neglikelihood, theta0, method=self.optimizer, bounds=bnds)
            if best_res is None or (res.success and res.fun < best_res.fun):
                best_res = res

        self.theta = best_res.x if best_res.success else theta0
        
        self.sigma2_noise_norm = 10**self.theta[0]
        self.length_scale_norm = 10**self.theta[2]
        self.Neglikelihood(self.theta)

    def predict(self, X_test):
        X_test_norm = (X_test - self.X_mean) / self.X_std
        signal_var = 10**self.theta[1]
        length_scale = 10**self.theta[2]
        
        k_s = self.Kernel(self.X_norm, X_test_norm, length_scale, signal_var)
        k_ss = self.Kernel(X_test_norm, X_test_norm, length_scale, signal_var)

        alpha = cho_solve((self.current_L, True), self.y_norm)
        mu_norm = k_s.T @ alpha
        
        v = cho_solve((self.current_L, True), k_s)
        cov_norm = k_ss - k_s.T @ v
        
        # Predictive variance (diagonal only) for standard plotting
        var_norm = np.diag(cov_norm) + self.sigma2_noise_norm

        mu = (mu_norm * self.y_std) + self.y_mean
        var = var_norm * (self.y_std**2)
        
        return mu.flatten(), var.flatten()

    def sample_posterior(self, X_test, n_samples=3):
        """
        Generates random function realizations from the posterior.
        """
        # 1. Normalize
        X_test_norm = (X_test - self.X_mean) / self.X_std
        
        # 2. Recompute Kernel Matrices
        signal_var = 10**self.theta[1]
        length_scale = 10**self.theta[2]
        
        k_s = self.Kernel(self.X_norm, X_test_norm, length_scale, signal_var)
        k_ss = self.Kernel(X_test_norm, X_test_norm, length_scale, signal_var)
        
        # 3. Posterior Mean (Normalized)
        alpha = cho_solve((self.current_L, True), self.y_norm)
        mu_norm = k_s.T @ alpha
        
        # 4. Posterior Covariance (Normalized)
        v = cho_solve((self.current_L, True), k_s)
        cov_norm = k_ss - k_s.T @ v
        
        # 5. Sample from Multivariate Normal (using Cholesky)
        # We need a stable Cholesky of the posterior covariance
        # The posterior covariance can be numerically unstable, so we add jitter
        n_test = X_test.shape[0]
        cov_norm += np.eye(n_test) * (1e-6) 
        
        try:
            L_post = np.linalg.cholesky(cov_norm)
        except np.linalg.LinAlgError:
            # Fallback if numerical issues arise
            L_post = np.linalg.cholesky(cov_norm + np.eye(n_test) * 1e-4)

        # Standard Normal Noise
        z = np.random.normal(0, 1, size=(n_test, n_samples))
        
        # Transform: mu + L * z
        f_samples_norm = mu_norm + L_post @ z
        
        # 6. Denormalize
        f_samples = (f_samples_norm * self.y_std) + self.y_mean
        
        return f_samples

# ==========================================
# 3. EXPERIMENT LOOP
# ==========================================
def main():
    x_min, x_max = 0.0, 1400.0
    test_function = TestFunction(1.49, 13.62, x_min, x_max, 0, 50, 0.2)
    
    # Initial random samples
    X_init_x = np.random.uniform(x_min, x_max, 3)
    X_curr = np.hstack([X_init_x.reshape(-1, 1), np.zeros((3, 1))])
    df_init = pd.DataFrame(X_curr, columns=['x', 'y'])
    y_curr = test_function.sample(df_init)['salinity'].values.reshape(-1, 1)

    # Plot grid
    x_grid = np.linspace(x_min, x_max, 400).reshape(-1, 1)
    y_true_grid = test_function.analytical_function(x_grid, 0).flatten()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(8):
        gp = GaussianProcess(kernel='matern32', n_restarts=10)
        gp.fit(X_curr[:, 0:1], y_curr)

        # Standard Prediction
        mu, var = gp.predict(x_grid)
        sigma = np.sqrt(var)
        
        # NEW: Sample Random Functions
        samples = gp.sample_posterior(x_grid, n_samples=5)

        ax = axes[i]
        
        # 1. Plot Truth
        ax.plot(x_grid, y_true_grid, 'k--', linewidth=1.5, label="True", zorder=3)
        
        # 2. Plot Samples (The "Converging Functions")
        # We plot these faintly behind the main lines
        ax.plot(x_grid, samples, color='teal', alpha=0.4, linewidth=0.8)
        
        # 3. Plot Mean and Confidence
        ax.plot(x_grid, mu, 'b-', linewidth=2, label="GP Mean", zorder=4)
        ax.fill_between(x_grid.flatten(), mu - 1.96*sigma, mu + 1.96*sigma, color='blue', alpha=0.15)
        
        # 4. Plot Data Points
        ax.scatter(X_curr[:,0], y_curr, c='r', s=40, zorder=5, edgecolors='white')
        
        ax.set_title(f"Iter {i} (l={gp.length_scale_norm:.2f}, samples=5)")
        ax.set_ylim(-5, 60) # Keep y-axis fixed to see convergence better

        if i < 7:
            _, var_grid = gp.predict(x_grid)
            next_idx = np.argmax(var_grid)
            next_x = x_grid[next_idx]
            
            X_new = np.array([[next_x[0], 0]])
            df_new = pd.DataFrame(X_new, columns=['x', 'y'])
            y_new = test_function.sample(df_new)['salinity'].values.reshape(-1, 1)
            
            X_curr = np.vstack([X_curr, X_new])
            y_curr = np.vstack([y_curr, y_new])
            
            ax.axvline(next_x, color='g', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()