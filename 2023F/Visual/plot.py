import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("2023F/Utils")
from utils_function import mean_2d

df = pd.read_csv("Dataset/radar_path.csv")
rain_id_unique_list = df["rain_id"].unique().tolist()
height_unique_list = df["height"].unique().tolist()
index_category_unique_list = df["index_category"].unique().tolist()


norm_params = {"dBZ": [0, 65], "ZDR": [-1, 5], "KDP": [-1, 6]}

for _, rain_id in enumerate(rain_id_unique_list):
    df_rain_id = df[df["rain_id"] == rain_id]
    print(rain_id)
    for _, index_category in enumerate(index_category_unique_list):
        df_index_category = df_rain_id[df_rain_id["index_category"] == index_category]
        for _, height in enumerate(height_unique_list):
            df_height = df_index_category[df_index_category["height"] == height]
            frame_id_list = df_height["frame_id"].tolist()
            radar_path_list = df_height["radar_path"].tolist()
            for i, radar_path in enumerate(radar_path_list):
                min_norm, max_norm = norm_params[index_category]
                radar_data = np.load(radar_path)
                radar_data = (radar_data - min_norm)/(max_norm - min_norm)
                plt.imshow(radar_data, cmap='coolwarm', interpolation='nearest')
                plt.colorbar()
                os.makedirs(f"2023F/Plotfigs/{rain_id}/{index_category}/{height}", exist_ok=True)
                plt.savefig(f"2023F/Plotfigs/{rain_id}/{index_category}/{height}/{frame_id_list[i]}.jpg")
                plt.close()
                
