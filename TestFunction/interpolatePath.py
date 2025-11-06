"""
Creates path for coordinate transformation
"""
    
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import pickle
from transformer import path_transform as transform

class InterpolatePath:
    def __init__(self, pointPath):
        self.df = pd.read_csv(pointPath)
        self.x, self.y = transform(self.df)
        # Initialize splines in constructor
        self.create_splines()  # Add this line

    def create_splines(self):
        """Creates CubicSpline interpolation functions for x and y coordinates--> this will be used as reference
        for 2nd transformation. Has C2 continuity.
        Args:
            pointPath: Path to the CSV file containing the points with 'x' and 'y' columns.
        Returns:
            None: Initializes self.xCord and self.yCord as CubicSpline objects.
        """
        data_index = np.arange(len(self.x))
        self.xCord = CubicSpline(data_index, self.x)  # have to make it a function of index or else it wouldn't be a function because it overlaps
        self.yCord = CubicSpline(data_index, self.y)
        index_smooth_cs = np.linspace(data_index.min(), data_index.max(), 1000)
        self.xCord = self.xCord(index_smooth_cs)
        self.yCord = self.yCord(index_smooth_cs)
 # Using forward slashes instead
    def save_path_object(self, filename):
        """Save the entire path object using pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Path object saved to {filename}")

    @staticmethod
    def load_path_object(filename):
        """Load a previously saved path object"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    path = InterpolatePath("data/path2.csv")
    path.save_path_object("transform_function_path.pkl")


