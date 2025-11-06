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
  origin_lon = -123.705492
  origin_lat = 48.388598

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

def integrate_arc_length(t, path):
  """Calculates the magnitude of the tangent vector at a given t.

  Args:
    t: The index on the spline.
    path: The InterpolatePath object containing the splines.

  Returns:
    The magnitude of the tangent vector at t.
  """
  # Calculate numerical derivatives
  t_array = np.arange(len(path.xCord))
  dx = np.interp(t, t_array, np.gradient(path.xCord, t_array))
  dy = np.interp(t, t_array, np.gradient(path.yCord, t_array))
  magnitude = np.sqrt(dx**2 + dy**2)
  return magnitude

def calculate_arc_length(t_end, path):
  #instead of recalculating the arc length each time from 0 to t_end, we can store the cumulative arc length
  
  """Calculates the arc length along the spline from t=0 to t_end.

  Args:
    t_end: The upper limit of integration for the arc length calculation.
    path: The InterpolatePath object containing the splines.

  Returns:
    The calculated arc length.
  """
  result, _ = quad(lambda t: integrate_arc_length(t, path), 0, t_end)
  return result
#how do I save th

