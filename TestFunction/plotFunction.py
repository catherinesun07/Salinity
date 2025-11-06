import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import matplotlib.cm as cm 
import matplotlib.colors as mcolors # Import matplotlib.colors as mcolors
import numpy as np
"Collection of plotting functions"

def plotPath(path):
    plt.figure(figsize=(10, 6))
    plt.plot(path.xCord, path.yCord, label='Interpolated Path', color='green')
    plt.scatter(path.x, path.y, label='Original Data', color='blue', alpha=0.6)
    plt.xlim(-100, 900)
    plt.ylim(-1200, 200)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Path Interpolation')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/path_plot.png") 
    
def plotPath_and_points(path, points):
    plt.figure(figsize=(10, 6))
    plt.plot(path.xCord, path.yCord, label='Continuous Path (CubicSpline)', color='green')
    plt.scatter(points['x'], points['y'], label='Test Points', color='red', alpha=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-100, 900)
    plt.ylim(-1200, 200)
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/test_points_on_path.png")

# ...existing code...

def plot_path_with_vectors(path, df, scale_factor=50):
    """
    Plot path with normal vectors and vectors to points
    Args:
        path: InterpolatePath object containing the path
        df: DataFrame with points and their closest points on path
        scale_factor: Length scaling for the vectors (default=50)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the path
    plt.plot(path.xCord, path.yCord, label='Path', color='green', linewidth=2)
    
    # Plot original points
    plt.scatter(df['x'], df['y'], label='Test Points', color='red', alpha=0.8)
    
    # Plot closest points
    plt.scatter(df['closest_x'], df['closest_y'], label='Closest Points', color='blue', alpha=0.8)
    
    # Plot normal vectors and vectors to points
    for i in range(len(df)):
        # Get points
        closest_point = np.array([df.loc[i, 'closest_x'], df.loc[i, 'closest_y']])
        test_point = np.array([df.loc[i, 'x'], df.loc[i, 'y']])
        
        # Calculate normal vector
        t = df.loc[i, 'closest_point']
        dx_dt = np.interp(t, np.arange(len(path.xCord)), np.gradient(path.xCord))
        dy_dt = np.interp(t, np.arange(len(path.yCord)), np.gradient(path.yCord))
        magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        normal = np.array([-dy_dt/magnitude, dx_dt/magnitude]) * scale_factor
        
        # Plot normal vector
        plt.arrow(closest_point[0], closest_point[1], 
                 normal[0], normal[1],
                 color='purple', alpha=0.5,
                 head_width=5, head_length=10,
                 label='Normal Vector' if i==0 else "")
        
        # Plot vector to point
        vector_to_point = test_point - closest_point
        plt.arrow(closest_point[0], closest_point[1],
                 vector_to_point[0], vector_to_point[1],
                 color='orange', alpha=0.5,
                 head_width=5, head_length=10,
                 label='Vector to Point' if i==0 else "")
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2nd transform')
    plt.legend()
    plt.grid(True)
    plt.xlim(-100, 900)
    plt.ylim(-1200, 200)
    plt.savefig("plots/path_with_vectors.png")
    plt.close()