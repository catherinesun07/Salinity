# gp_start_with_saved_testfunc.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor  # GP model[web:118]
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # kernels[web:118]

# contain an object with a callable interface:
#   y = test_func.predict(x, y) or y = test_func(X) where X = [[x,y], ...].
def load_test_function(pkl_rel_path):
    here = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(here, pkl_rel_path)
    # Compatibility shim: some pickles were created when the class
    # (SmoothSplineInterpolator) was defined in a script run as
    # "__main__". When that happens, pickle records the class as
    # __main__.SmoothSplineInterpolator which fails to import here.
    # To support those pickles, import the original module and inject
    # the class object into the current __main__ module before
    # unpickling so the class can be resolved.
    try:
        import TestFunction
        import __main__ as _main_mod
        for _name in ("SmoothSplineInterpolator",):
            if hasattr(TestFunction, _name) and not hasattr(_main_mod, _name):
                setattr(_main_mod, _name, getattr(TestFunction, _name))
    except Exception:
        # If TestFunction cannot be imported, proceed and let pickle
        # raise its own error (which will be more informative).
        pass

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def evaluate_test_function(test_func, X):
    x = X[:, 0]
    y = X[:, 1]
    # Try y = f.predict(x, y)
    if hasattr(test_func, "predict"):
        try:
            out = test_func.predict(x, y)
            return np.asarray(out).reshape(-1)
        except Exception:
            try:
                out = test_func.predict(X)
                return np.asarray(out).reshape(-1)
            except Exception:
                pass
    # Try callable interface
    try:
        out = test_func(X)
        return np.asarray(out).reshape(-1)
    except Exception as e:
        raise TypeError(
            "Could not evaluate the loaded test function with predict(x,y), predict(X), or call(X)."
        ) from e

TEST_FUNC_PKL = r"models\TestFunction_20251104-124533.pkl"  # adjust if needed
test_func = load_test_function(TEST_FUNC_PKL)  # loads saved test funciton

#  10 sensor locations and get measurements function -----
rng = np.random.default_rng(0)
# specifying the domain since the function is unbounded: x in [0, 200], y in [0, 300] (adjust to your coordinate system)
X_train = rng.uniform([0, 0], [200, 300], size=(10, 2))  # 10 initial sensors
y_train = evaluate_test_function(test_func, X_train)  # evaluate saved function at sensor sites

# addingnoise to emulate measurement error, I'm not sure if this is needed 
y_train_noisy = y_train + rng.normal(0, 0.01, size=y_train.shape)  # small Gaussian noise

#building  and fitt the GP model -----
# anisotropic RBF for x/y with learnable scales, plus a noise term; normalized outputs help optimization
kernel = ConstantKernel(1.0) * RBF(length_scale=[80.0, 120.0], length_scale_bounds=(1e-1, 1e3)) \
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))  # reasonable starting priors
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8, normalize_y=True)  # GP regressor
gp.fit(X_train, y_train_noisy)  # hyperparameters tuned by marginal likelihood

#predict on a grid for visualization -----
xg = np.linspace(0, 200, 120)  # adjust resolution as needed
yg = np.linspace(0, 300, 180)  # adjust resolution as needed
Xg, Yg = np.meshgrid(xg, yg)
X_grid = np.column_stack([Xg.ravel(), Yg.ravel()])

mean, std = gp.predict(X_grid, return_std=True)  # predictive mean and uncertainty
mean = mean.reshape(Yg.shape)
std = std.reshape(Yg.shape)

#plotting mean and uncertainty with sensor points -----
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

im0 = axes[0].pcolormesh(Xg, Yg, mean, cmap="viridis", shading="auto")  # mean surface
axes[0].scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="black", s=60, label="Sensors")  # sensors
axes[0].set_title("GP Predicted Mean")  # title
axes[0].set_xlabel("x")  # x label
axes[0].set_ylabel("y")  # y label
axes[0].set_aspect("equal")  # equal aspect
axes[0].legend()  # legend
plt.colorbar(im0, ax=axes[0], label="Salinity")  # colorbar

im1 = axes[1].pcolormesh(Xg, Yg, std, cmap="magma", shading="auto")  # std surface
axes[1].scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="black", s=60, label="Sensors")  # sensors
axes[1].set_title("GP Predictive Std (Uncertainty)")  # title
axes[1].set_xlabel("x")  # x label
axes[1].set_ylabel("y")  # y label
axes[1].set_aspect("equal")  # equal aspect
axes[1].legend()  # legend
plt.colorbar(im1, ax=axes[1], label="Std")  # colorbar

plt.show()  # display plots

#prediction/suggesting for next points 
idx = np.argmax(std)  # index of max std in grid
next_xy = np.array([Xg.ravel()[idx], Yg.ravel()[idx]])  # suggested next sensor location
print("Suggested next sensor location (max-uncertainty):", next_xy)  # report candidate
