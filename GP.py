import numpy as np
import matplotlib.pyplot as plt
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
# 2. THE BRAIN (Fixed Gaussian Process)
# ==========================================
class GaussianProcess:
    def __init__(self, kernel='matern32', length_scale=0.83, signal_var=1.0, noise_var=0.1, jitter=1e-6):
        self.kernel = kernel
        self.jitter = jitter
        self.theta = None
        
        # FIXED HYPERPARAMETERS (User defined)
        self.fixed_length_scale = length_scale
        self.fixed_signal_var = signal_var
        self.fixed_noise_var = noise_var
        
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

    def compute_matrices(self):
        """
        Computes the Kernel matrix K and Cholesky decomposition L
        based on the FIXED hyperparameters and current data.
        """
        # Unpack fixed parameters
        sigma2_noise = self.fixed_noise_var
        signal_var = self.fixed_signal_var
        length_scale = self.fixed_length_scale

        # Store theta for consistency with predict methods (log10 format)
        self.theta = np.log10([sigma2_noise, signal_var, length_scale])

        n = self.X_norm.shape[0]
        K = self.Kernel(self.X_norm, self.X_norm, length_scale, signal_var) 
        K += np.eye(n) * (sigma2_noise + self.jitter)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fallback for numerical stability if needed
            L = np.linalg.cholesky(K + np.eye(n) * 1e-5)

        self.current_K = K
        self.current_L = L

        # We also need alpha for predictions
        self.alpha = cho_solve((L, True), self.y_norm)

    def fit(self, X, y):
        # 1. Normalize Data
        self.X_mean, self.X_std = np.mean(X), np.std(X)
        if self.X_std < 1e-8: self.X_std = 1.0
        self.X_norm = (X - self.X_mean) / self.X_std

        self.y_mean, self.y_std = np.mean(y), np.std(y)
        if self.y_std < 1e-8: self.y_std = 1.0
        self.y_norm = (y - self.y_mean) / self.y_std

        # 2. NO OPTIMIZATION. Just compute matrices with fixed params.
        self.compute_matrices()
        
        # Save explicit norm values for plotting/debug
        self.sigma2_noise_norm = self.fixed_noise_var
        self.length_scale_norm = self.fixed_length_scale

    def predict(self, X_test):
        X_test_norm = (X_test - self.X_mean) / self.X_std
        
        # Retrieve fixed params
        signal_var = self.fixed_signal_var
        length_scale = self.fixed_length_scale
        
        k_s = self.Kernel(self.X_norm, X_test_norm, length_scale, signal_var)
        k_ss = self.Kernel(X_test_norm, X_test_norm, length_scale, signal_var)

        # Use precomputed alpha
        mu_norm = k_s.T @ self.alpha
        
        v = cho_solve((self.current_L, True), k_s)
        cov_norm = k_ss - k_s.T @ v
        
        # Predictive variance (diagonal only)
        var_norm = np.diag(cov_norm) + self.sigma2_noise_norm

        mu = (mu_norm * self.y_std) + self.y_mean
        var = var_norm * (self.y_std**2)
        
        return mu.flatten(), var.flatten()

    def sample_posterior(self, X_test, n_samples=3):
        # 1. Normalize
        X_test_norm = (X_test - self.X_mean) / self.X_std
        
        # 2. Use Fixed Params
        signal_var = self.fixed_signal_var
        length_scale = self.fixed_length_scale
        
        k_s = self.Kernel(self.X_norm, X_test_norm, length_scale, signal_var)
        k_ss = self.Kernel(X_test_norm, X_test_norm, length_scale, signal_var)
        
        # 3. Posterior Mean (Normalized)
        mu_norm = k_s.T @ self.alpha
        
        # 4. Posterior Covariance (Normalized)
        v = cho_solve((self.current_L, True), k_s)
        cov_norm = k_ss - k_s.T @ v
        
        # 5. Sample
        n_test = X_test.shape[0]
        cov_norm += np.eye(n_test) * (1e-6) 
        
        try:
            L_post = np.linalg.cholesky(cov_norm)
        except np.linalg.LinAlgError:
            L_post = np.linalg.cholesky(cov_norm + np.eye(n_test) * 1e-4)

        z = np.random.normal(0, 1, size=(n_test, n_samples))
        f_samples_norm = mu_norm + L_post @ z
        
        # 6. Denormalize
        f_samples = (f_samples_norm * self.y_std) + self.y_mean
        
        return f_samples

# ==========================================
# 3. EXPERIMENT LOOP
# ==========================================
def main():
    x_min, x_max = 0.0, 1400.0
    test_function = TestFunction(1.49, 13.62, x_min, x_max, 0, 50, 0.9)
    
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

    # Define the FIXED hyperparameters here
    # L=0.83 as requested. 
    # (Note: This is applied to normalized data, which is standard for ML GPs)
    fixed_gp_params = {
        'length_scale': 0.83,
        'signal_var': 1.0,
        'noise_var': 0.05
    }

    for i in range(8):
        # Pass fixed params into constructor
        gp = GaussianProcess(kernel='matern32', **fixed_gp_params)
        
        # Fit strictly calculates posterior, no tuning
        gp.fit(X_curr[:, 0:1], y_curr)

        # Standard Prediction
        mu, var = gp.predict(x_grid)
        sigma = np.sqrt(var)
        
        # Sample Random Functions
        samples = gp.sample_posterior(x_grid, n_samples=5)

        ax = axes[i]
        
        # 1. Plot Truth
        ax.plot(x_grid, y_true_grid, 'k--', linewidth=1.5, label="True", zorder=3)
        
        # 2. Plot Samples
        ax.plot(x_grid, samples, color='teal', alpha=0.4, linewidth=0.8)
        
        # 3. Plot Mean and Confidence
        ax.plot(x_grid, mu, 'b-', linewidth=2, label="GP Mean", zorder=4)
        ax.fill_between(x_grid.flatten(), mu - 1.96*sigma, mu + 1.96*sigma, color='blue', alpha=0.15)
        
        # 4. Plot Data Points
        ax.scatter(X_curr[:,0], y_curr, c='r', s=40, zorder=5, edgecolors='white')
        
        ax.set_ylim(-5, 60)

        if i < 7:
            # Active Learning (Variance Reduction)
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