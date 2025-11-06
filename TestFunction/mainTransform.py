from interpolatePath import InterpolatePath
import transformer
import pandas as pd
import numpy as np
import plotFunction
from scipy.optimize import minimize

DATA_PATH = "data/Sensors-merged.csv"
FUNC_PATH = "transform_function_path.pkl"


def loadData(filePath):
    df = pd.read_csv(filePath)
    x, y = transformer.path_transform(df)
    df['x'] = x
    df['y'] = y
    return df

#how do I save the function path?
if __name__ == "__main__":
    df = loadData(DATA_PATH)
    path = InterpolatePath.load_path_object(FUNC_PATH)
    
    #visualizing path and plots
    plotFunction.plotPath(path)
    plotFunction.plotPath_and_points(path, df)

    # Create new DataFrame with selected columns
    transformed_points = df[['salinity']].copy()
    # Initialize new columns

    
    for i in range(len(df)):
        point = (df['x'].iloc[i], df['y'].iloc[i])
        # Add bounds to keep t within the valid range
        bounds = [(0, len(path.xCord) - 1)]
        result = minimize(transformer.L2, x0=len(path.xCord)/2, 
                        args=(point, path), method='L-BFGS-B', bounds=bounds)
        #df.loc[i, 'x_prime'] = transformer.calculate_arc_length(result.x[0], path)
        #df.loc[i, 'closest_point'] = result.x[0] # Use .loc for assignment
        
        x_prime = transformer.calculate_arc_length(result.x[0], path)
        transformed_points.loc[i, 'x_prime'] = x_prime#adding transformed x coordinate to new dataframe
        closest_point = result.x[0]
        
        """
        Your path is interpolated into 1000 points
        The closest point is around point 76 out of these 1000 points
        More precisely, it's about 71% between points 75 and 76 of the 1000 interpolated points
        """
        #df.loc[i, 'closest_x'] = np.interp(result.x[0], np.arange(len(path.xCord)), path.xCord)#x cordinate of closest point on spline
        #df.loc[i, 'closest_y'] = np.interp(result.x[0], np.arange(len(path.yCord)), path.yCord)#y cordinate of closest point on spline
        closest_x = np.interp(result.x[0], np.arange(len(path.xCord)), path.xCord)
        closest_y = np.interp(result.x[0], np.arange(len(path.yCord)), path.yCord)
    
  #finding normal vector at each closest point on path
        t = closest_point
        # Calculate derivatives
        dx_dt = np.interp(t, np.arange(len(path.xCord)), np.gradient(path.xCord))
        dy_dt = np.interp(t, np.arange(len(path.yCord)), np.gradient(path.yCord))
        # Calculate magnitude
        magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        # Normalize to get unit vector
        normal_x = -dy_dt / magnitude
        normal_y = dx_dt / magnitude

        path_normal = np.array([normal_x, normal_y])
        direction_vector = np.array([df.loc[i, 'x'] - closest_x, df.loc[i, 'y'] - closest_y])
        
        y_prime = np.dot(path_normal, direction_vector)
        
        transformed_points.loc[i, 'y_prime'] = np.linalg.norm(y_prime)#finds the normal distance of point from curve
    
    
    #df.to_csv("data/test_with_closest_points.csv", index=False)
    #print(df)
    transformed_points.to_csv("data/transformed_points.csv", index=False)
    #print(transformed_points)
    #plotFunction.plot_path_with_vectors(path, df)
        
        
        
        
