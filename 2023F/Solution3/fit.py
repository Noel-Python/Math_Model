import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


frame_id = "frame_121.npy"
rain_id = "data_dir_062"

rain = np.load(f"Dataset/NJU_CPOL_kdpRain/{rain_id}/{frame_id}")
zh = np.load(f"Dataset/NJU_CPOL_update2308/dBZ/3.0km/{rain_id}/{frame_id}")
zdr = np.load(f"Dataset/NJU_CPOL_update2308/ZDR/3.0km/{rain_id}/{frame_id}")

plt.imshow(rain, cmap='rainbow', interpolation='nearest')
plt.colorbar()
plt.savefig(f"real.jpg")
plt.close()

for i in range(rain.shape[0]):
    for j in range(rain.shape[1]):
        rain[i][j] = (0.0085 * (zh[i][j]**0.896) * (zdr[i][j]**(-0.2877))) + 200

plt.imshow(rain, cmap='rainbow', interpolation='nearest')
plt.colorbar()
plt.savefig(f"fake.jpg")
plt.close()