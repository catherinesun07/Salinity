import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import matplotlib.cm as cm 
import matplotlib.colors as mcolors # Import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def plotPoints(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x_prime'], df['salinity'], label='Test Points', color='blue', alpha=0.8)
    plt.xlabel('x(distance down estuary)')
    plt.ylabel('Salinity')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/transformed_points_plot.png")


if __name__ == "__main__":
    df = pd.read_csv("data/transformed_points.csv")
    plotPoints(df)
    
    

