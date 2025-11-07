#transforming cordinate system
from pyproj import Transformer
import numpy as np # Ensure numpy is imported
from scipy.integrate import quad


def path_transform(df):
  """Transforms geographic coordinates to a local Cartesian system centered at a fixed origin.

  Args:
    df: The input DataFrame containing 'longitude' and 'latitude' columns.

  Returns:
    A tuple of (x, y) coordinates in the local Cartesian system.
  """
  transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

  # the fixed origin coordinates
  origin_lon = -123.70539093017600
  origin_lat = 48.38793182373050

  # transforming the fixed origin coordinates
  origin_e, origin_n = transformer.transform(origin_lon, origin_lat)

  # transforming the input DataFrame coordinates
  e, n = transformer.transform(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())

  # center to (0,0) at the fixed origin, will be important for the second transform
  df["x"] = e - origin_e
  df["y"] = n - origin_n

  return df["x"], df["y"]

def L2(t, point, path):
  """Calculates the Euclidean distance between a point on the spline at index t and a given point.

  Args:
    t: The index on the spline.
    point: A tuple or list representing the coordinates of the given point (x, y).
    path: The InterpolatePath object containing the splines.

  Returns:
    The Euclidean distance between the interpolated point and the given point.
  """
  # Calculate interpolated point on the spline
  x_t = np.interp(t, np.arange(len(path.xCord)), path.xCord)
  y_t = np.interp(t, np.arange(len(path.yCord)), path.yCord)
  
  # Calculate distance to the test point
  distance = np.sqrt((x_t - point[0])**2 + (y_t - point[1])**2)
  return distance

def calculate_cumulative_arclengths(path):
    """Precalculate cumulative arc lengths for the entire path using trapezoidal rule.
    
    Args:
        path: The InterpolatePath object containing the splines.
    
    Returns:
        numpy array of cumulative arc lengths at each point
    """
    # Get equally spaced points along the path
    t_array = np.arange(len(path.xCord))
    
    # Calculate derivatives
    dx = np.gradient(path.xCord, t_array)
    dy = np.gradient(path.yCord, t_array)
    
    # Calculate segment lengths
    segment_lengths = np.sqrt(dx**2 + dy**2)
    
    # Use trapezoidal rule for more accurate integration
    cumulative_lengths = np.zeros_like(t_array, dtype=np.float64)
    cumulative_lengths[1:] = np.cumsum(0.5 * (segment_lengths[:-1] + segment_lengths[1:]))
    
    return cumulative_lengths

def calculate_arc_length(t_end, path, cumulative_lengths=None):
    """Calculates the arc length along the spline from t=0 to t_end.
    
    Args:
        t_end: The parameter value to calculate length to
        path: The InterpolatePath object
        cumulative_lengths: Pre-calculated cumulative lengths
        
    Returns:
        The arc length to the given parameter value
    """
    if cumulative_lengths is None:
        # Calculate cumulative lengths if not provided
        cumulative_lengths = calculate_cumulative_arclengths(path)
    
    # Ensure t_end is within bounds
    t_end = np.clip(t_end, 0, len(path.xCord) - 1)
    
    # Use linear interpolation for sub-interval accuracy
    return np.interp(t_end, np.arange(len(path.xCord)), cumulative_lengths)
#how do I save th

