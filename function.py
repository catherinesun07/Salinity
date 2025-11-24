import numpy as np
import pandas as pd

class TestFunction:
    def __init__(self, a, b, x_max, y_max, x_min, y_min, noise_std):
        """
        Initializes the test function with noise level and boundaries.

        Args:
            noise_std_dev (float): The std of the guassian noise distribution. The mean 
            is assumed to be 0.
            x_min (float): The minimum allowed value for x.
            x_max (float): The maximum allowed value for x.
            y_min (float): The minimum allowed value for the noisy y.
            y_max (float): The maximum allowed value for the noisy y.
        """
        if not isinstance(noise_std, (int, float)) or noise_std< 0:
            raise ValueError("noise std must be a non negative")
        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max")
        if y_min >= y_max:
            raise ValueError("y_min must be less than y_max")
        
        self.a = a
        self.b = b
        self.y_max = y_max
        self.y_min = y_min
        self.x_max = x_max
        self.x_min = x_min
        self.noise_std = noise_std
    
    def analytical_function(self, x):
        """the true function without noise"""
        if np.any(x <= 0):
            raise ValueError("x coordinate must be positive for the natural logarithm.")
        return self.a * np.log(x) + self.b
    
    
    def sample(self, sample__points):
        """
        Samples points from the function, adds gaussian noise based on 
        defined noise_std, and filters out points outside the specified boundaries.
      
        Args:
            df_points (pd.DataFrame): A DataFrame with an 'x_prime' column of values to sample at.

        Returns:
            pd.DataFrame: A new DataFrame with 'x_prime' and 'salinity' columns
                          containing the points that are within the specified boundaries.
        """
        if not isinstance(sample__points, pd.DataFrame) or 'x' not in sample__points.columns:
            raise TypeError("Input must be a pandas DataFrame with an 'x' column.")

        x= sample__points['x'].values
        y = sample__points['y'].values
        
        noise = np.random.normal(0, self.noise_std)

        salinity = self.analytical_function(x) + noise
        if not (self.x_min <= x<= self.x_max and self.y_min <= y <= self.y_max):
        # Return NaN for salinity if the sample is rejected
            return np.nan
        sample__points['salinity'] = salinity
        
        return 
    
        
        
        
        

