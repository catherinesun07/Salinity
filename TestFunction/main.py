import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import matplotlib.cm as cm 
import matplotlib.colors as mcolors # Import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pickle

def plotPoints(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x_prime'], df['salinity'], label='Test Points', color='blue', alpha=0.8)
    plt.xlabel('x(distance down estuary)')
    plt.ylabel('Salinity')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/transformed_points_plot.png")
    
class LogModel:
    """Class to hold the logarithmic model and its parameters"""
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b
    
    def fit(self, df):
        """
        Fits a logarithmic function of form: y = a * log(x) + b
        Args:
            df: DataFrame with x_prime and salinity columns
        """
        # Remove any negative or zero x values as log is undefined there
        mask = df['x_prime'] > 0
        x = df.loc[mask, 'x_prime']
        y = df.loc[mask, 'salinity']
        
        # Fit logarithmic function using least squares
        A = np.vstack([np.log(x), np.ones(len(x))]).T
        self.a, self.b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Plot the fit
        self.plot_fit(x, y)
        
        return self
    
    def predict(self, x):
        """Apply the logarithmic function to input x"""
        if self.a is None or self.b is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.a * np.log(x) + self.b
    
    def plot_fit(self, x, y):
        """Plot the data and fitted curve"""
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Data Points')
        
        # Generate points for smooth curve
        x_smooth = np.linspace(x.min(), x.max(), 1000)
        y_smooth = self.predict(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r-', label=f'y = {self.a:.2f}ln(x) + {self.b:.2f}')
        
        plt.xlabel('Distance Down Estuary')
        plt.ylabel('Salinity')
        plt.title('Log Fit')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/log_fit.png')
        plt.close()
    
    def save_model(self, filename):
        """Save the model to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load_model(filename):
        """Load the model from a file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Load and plot the data
    df = pd.read_csv("data/transformed_points.csv")
    plotPoints(df)
    
    # Create and fit the model
    model = LogModel()
    model.fit(df)
    print(f"Fitted parameters: a = {model.a:.4f}, b = {model.b:.4f}")
    
    # Save the model
    model.save_model("test_function.pkl")
    
    # Example of loading and using the model
  
    
    