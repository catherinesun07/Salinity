from interpolatePath import InterpolatePath
import transformer
import pandas as pd
import numpy as np
import plotFunction
from scipy.optimize import minimize

DATA_PATH = "data/Sensors-merged.csv"
FUNC_PATH = "newpath.pkl"


def loadData(filePath):
    df = pd.read_csv(filePath)
    x, y = transformer.path_transform(df)
    df['x'] = x
    df['y'] = y
    return df

#how do I save the function path?
if __name__ == "__main__":
    df = loadData(DATA_PATH)
    
    # Filter out points where y is below threshold, keep all other columns
    Y_THRESHOLD = -1100  # Adjust this threshold value as needed
    print(f"Number of points before filtering: {len(df)}")
    print(f"Columns in dataset: {df.columns.tolist()}")
    
    # Keep all columns, just filter rows based on y value
    mask = df['y'] >= Y_THRESHOLD
    df = df[mask].copy()
    df = df.reset_index(drop=True)  # Reset index after filtering
    
    
    path = InterpolatePath.load_path_object(FUNC_PATH)
    
    #visualizing path and plots
    plotFunction.plotPath(path)
    plotFunction.plotPath_and_points(path, df)

    # Create new DataFrame with selected columns
    transformed_points = df[['salinity']].copy()
    # Initialize new columns for transformed coordinates
    transformed_points['x_prime'] = np.nan
    transformed_points['y_prime'] = np.nan
    
    # Pre-calculate cumulative arc lengths for efficiency
    cumulative_lengths = transformer.calculate_cumulative_arclengths(path)
    
    # Pre-calculate gradients for normal vectors
    t_array = np.arange(len(path.xCord))
    dx_dt_array = np.gradient(path.xCord)
    dy_dt_array = np.gradient(path.yCord)
    
    for i in range(len(df)):
        point = (df['x'].iloc[i], df['y'].iloc[i])
        # Add bounds to keep t within the valid range
        bounds = [(0, len(path.xCord) - 1)]
        result = minimize(transformer.L2, x0=len(path.xCord)/2, 
                        args=(point, path), method='L-BFGS-B', bounds=bounds)
        
        closest_point = result.x[0]
        # Calculate x_prime using pre-calculated cumulative lengths
        x_prime = transformer.calculate_arc_length(closest_point, path, cumulative_lengths)
        transformed_points.loc[i, 'x_prime'] = np.round(x_prime, decimals=6)  # Round to 6 decimal places
        
        """
        Your path is interpolated into 1000 points
        The closest point is around point 76 out of these 1000 points
        More precisely, it's about 71% between points 75 and 76 of the 1000 interpolated points
        """
        #df.loc[i, 'closest_x'] = np.interp(result.x[0], np.arange(len(path.xCord)), path.xCord)#x cordinate of closest point on spline
        #df.loc[i, 'closest_y'] = np.interp(result.x[0], np.arange(len(path.yCord)), path.yCord)#y cordinate of closest point on spline
        closest_x = np.interp(result.x[0], np.arange(len(path.xCord)), path.xCord)
        closest_y = np.interp(result.x[0], np.arange(len(path.yCord)), path.yCord)
    
        # Calculate normal vector using pre-calculated gradients
        dx_dt = np.interp(closest_point, t_array, dx_dt_array)
        dy_dt = np.interp(closest_point, t_array, dy_dt_array)
        magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # Normalize to get unit vector
        normal_x = -dy_dt / magnitude
        normal_y = dx_dt / magnitude
        path_normal = np.array([normal_x, normal_y])
        
        # Calculate direction vector
        direction_vector = np.array([df.loc[i, 'x'] - closest_x, df.loc[i, 'y'] - closest_y])
        
        # Calculate y_prime (signed distance)
        y_prime = np.dot(path_normal, direction_vector)
        
        # Store rounded value
        transformed_points.loc[i, 'y_prime'] = np.round(y_prime, decimals=6)
    
    
    #df.to_csv("data/test_with_closest_points.csv", index=False)
    #print(df)
    transformed_points.to_csv("data/transformed_points.csv", index=False)
    #print(transformed_points)
    #plotFunction.plot_path_with_vectors(path, df)
        
        
        
        
