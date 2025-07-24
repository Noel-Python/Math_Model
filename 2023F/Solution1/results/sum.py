import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import polynomial as P


df1_1km = pd.read_csv(f"2023F/Solution1/results/1km_35.csv")
df2_1km = pd.read_csv(f"2023F/Solution2/results/1km_35.csv")
df1_3km = pd.read_csv(f"2023F/Solution1/results/3km_35.csv")
df2_3km = pd.read_csv(f"2023F/Solution2/results/3km_35.csv")
df1_7km = pd.read_csv(f"2023F/Solution1/results/7km_35.csv")
df2_7km = pd.read_csv(f"2023F/Solution2/results/7km_35.csv")

x = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60])

plt.figure(figsize=(5, 4), dpi=150)
p = P.polynomial.Polynomial.fit(x, df1_1km["mse"], 5)
plt.plot(x, p(x))

plt.plot(x, df2_1km["mse"])
plt.plot(x, df1_3km["mse"])
plt.plot(x, df2_3km["mse"])
plt.plot(x, df1_7km["mse"])
plt.plot(x, df2_7km["mse"])
plt.show()
plt.legend()
plt.savefig("sum.png")
plt.close()