import numpy as np
import pandas as pd

class TestFunction:
    def __init__(self, a, c, x_min, x_max, y_min, y_max, noise_std):
        """
        Initializes the test function which models salinity.
        The function takes 2D coordinates (x, y) but salinity only depends on x.
        Salinity = a * log(x) + c + noise

        Args:
            a, c (float): Coefficients for the analytical function.
            x_min, x_max (float): Boundaries for the x-coordinate.
            y_min, y_max (float): Boundaries for the y-coordinate.
            noise_std (float): Standard deviation of Gaussian noise.
        """
        self.a = a
        self.c = c
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.noise_std = noise_std

    def analytical_function(self, x, y):
        """
        Analytical form of the function.
        Returns NaN if x or y are out of bounds.
        """
        # Vectorized boundary check
        in_bounds = (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)
        
        # Calculate salinity only where in bounds
        safe_x = np.maximum(x, 1e-9)
        salinity = self.a * np.log(safe_x) + self.c
        
        # Return salinity or NaN
        return np.where(in_bounds, salinity, np.nan)

    def sample(self, sample_points):
        """
        Generates noisy salinity samples for a given set of coordinates.
        
        Args:
            sample_points (pd.DataFrame): DataFrame with 'x' and 'y' columns.
                                          
        Returns:
            pd.DataFrame: DataFrame with 'x', 'y', and 'salinity' columns.
                          Salinity is NaN for out-of-bounds points.
        """
        if not isinstance(sample_points, pd.DataFrame) or not {'x', 'y'}.issubset(sample_points.columns):
            raise ValueError("sample_points must be a pandas DataFrame with 'x' and 'y' columns.")

        x_vals = sample_points['x'].values
        y_vals = sample_points['y'].values

        # Get true values (with potential NaNs for out-of-bounds)
        true_salinity = self.analytical_function(x_vals, y_vals)
        
        # Add noise only to the valid points
        noise = np.random.normal(0, self.noise_std, size=true_salinity.shape)
        noisy_salinity = np.where(np.isnan(true_salinity), np.nan, true_salinity + noise)
        
        result_df = sample_points.copy()
        result_df['salinity'] = noisy_salinity
        
        return result_df

if __name__ == "__main__":
    # Example usage
    test_function = TestFunction(a=1.49, c=13.62, x_min=0.0, x_max=1400.0, y_min=0.0, y_max=50.0, noise_std=0.8)
    sample_df = pd.DataFrame({'x': [10, 500, 1500], 'y': [10, 25, 60]})
    sampled_data = test_function.sample(sample_df)
    print(sampled_data)