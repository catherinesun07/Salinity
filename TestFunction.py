#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 
from scipy.interpolate import UnivariateSpline
from pyproj import Transformer

CSV_NAME = "Sensors-merged.csv"           # must be in same folder
OUTPUT_PKL = "salinity_interpolator.pkl"  # saved model name

def load_and_prepare_same_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, CSV_NAME)


    df = pd.read_csv(csv_path)

    # column cleanup and spatial crop (adapt as needed)
    if "temperature" in df.columns:
        df = df.drop("temperature", axis=1)

    df = df[(df["Longitude"] > -123.708391) & (df["Longitude"] <= -123.692)]
    df = df[(df["Latitude"] >= 48.37800210) & (df["Latitude"] < 48.39068470053587)]

    if "salinity" not in df.columns:
        raise ValueError("Expected a 'salinity' column in the CSV.")

    # lon/lat -> UTM (EPSG:32610), then center
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    e, n = transformer.transform(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
    e0, n0 = np.median(e), np.median(n)
    df["x"] = e - e0
    df["y"] = n - n0

    # sort by y and de-duplicate for spline stability
    df = df.sort_values("y").drop_duplicates(subset=["y"], keep="first").reset_index(drop=True)

    return df[["x", "y", "salinity"]]

class SmoothSplineInterpolator:
    """
    Spline along y; x is ignored so the field varies smoothly with y only.
    """

    def __init__(self, y, salinity, window_size=5, smoothing_factor=None):
        #  average for smoothing 
        sal_ma = pd.Series(salinity).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

        # auto smoothing based on variance if smoothing factor is not provided
        if smoothing_factor is None:
            smoothing_factor = len(y) * np.var(sal_ma) * 0.01

        self.y = np.asarray(y)
        self.sal_raw = np.asarray(salinity)
        self.sal_ma = sal_ma
        self.s = float(smoothing_factor)
        self.spline = UnivariateSpline(self.y, self.sal_ma, s=self.s, k=3)

    def predict(self, x, y):
        return self.spline(y)

    def grid(self, x_range=None, y_range=None, resolution=100):
        if x_range is None:
            x_range = (-400, 400)
        if y_range is None:
            y_range = (float(self.y.min()), float(self.y.max()))
        xg = np.linspace(x_range[0], x_range[1], resolution)
        yg = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(xg, yg)
        Z = self.spline(Y)
        return X, Y, Z

    def plot_heatmap(self, x_range=None, y_range=None, resolution=100, show_points=False):
        X, Y, Z = self.grid(x_range, y_range, resolution)
        fig, ax = plt.subplots(figsize=(24, 16))
        im = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
        plt.colorbar(im, ax=ax, label="Salinity")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Smoothed Salinity Heatmap (s={self.s:.3f})")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

def main():
    df = load_and_prepare_same_folder()

    # interpolator settings
    interp = SmoothSplineInterpolator(
        y=df["y"].values,
        salinity=df["salinity"].values,
        window_size=31,
        smoothing_factor=None,  # auto
    )

    interp.plot_heatmap()

    #saving the model
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")  # [create path][web:91]
    os.makedirs(models_dir, exist_ok=True)  # [ensure folder exists][web:91]
    
    ts = datetime.now().strftime("%Y%m%d-%H%M%S") 
    filename = f"TestFunction_{ts}.pkl"
    out_path = os.path.join(models_dir, filename)  # [target file path][web:91]
    with open(out_path, "wb") as f:
        pickle.dump(interp, f)  # [save pickle][file:47]
    print(f"Saved interpolator to: {out_path}")  # [confirm path][file:47py
   

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, OUTPUT_PKL)
    with open(out_path, "wb") as f:
        pickle.dump(interp, f)
    print(f"Saved interpolator to: {out_path}")

if __name__ == "__main__":
    main()
