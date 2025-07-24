import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import sys

# 剔除数据集中非降雨样本
def kill_no_rain():
    I_threshold = 35.0
    A_threshold = 256.0
    no_rain_id_list = []
    df = pd.read_csv("Dataset/q1_radar_raw_dataset.csv")
    
    df_1km = df[df["height"] == "1.0km"]
    df_1km_dBZ = df_1km[df_1km["index_category"] == "dBZ"]
    count = 0
    for _, radar_data in enumerate(df_1km_dBZ["output"]):
        max_A = 0.0
        A = 0.0
        for radar_path in eval(radar_data):
            radar_path_data = np.load(radar_path)
            A = np.sum(radar_path_data >= I_threshold)
            add_A = np.sum(radar_path_data >= I_threshold)
            A += add_A
            max_A = max_A if max_A>A else A 
        if max_A < A_threshold:
            count += 1
    print(f"{count} / {len(df_1km_dBZ)}") 
    
    df_3km = df[df["height"] == "7.0km"]
    df_3km_dBZ = df_3km[df_3km["index_category"] == "dBZ"]
    df_3km_dBZ_after_kill = df_3km_dBZ.copy()
    count = 0
    for _, radar_data in enumerate(df_3km_dBZ["output"]):
        max_A = 0.0
        A = 0.0
        for radar_path in eval(radar_data):
            radar_path_data = np.load(radar_path)
            A = np.sum(radar_path_data >= I_threshold)
            add_A = np.sum(radar_path_data >= I_threshold)
            A += add_A
            max_A = max_A if max_A>A else A 
        if max_A < A_threshold:
            kill_index = df_3km_dBZ[df_3km_dBZ["output"] == radar_data].index
            df_3km_dBZ_after_kill.drop(index=kill_index, inplace=True)
            count += 1
    print(f"{count} : {len(df_3km_dBZ_after_kill)} / {len(df_3km_dBZ)} ") 
    df_3km_dBZ_after_kill.to_csv(f"Dataset/radar_7km_kill_norain.csv")
    
    df_7km = df[df["height"] == "7.0km"]
    df_7km_dBZ = df_7km[df_7km["index_category"] == "dBZ"]
    count = 0
    for _, radar_data in enumerate(df_7km_dBZ["output"]):
        max_A = 0.0
        A = 0.0
        for radar_path in eval(radar_data):
            radar_path_data = np.load(radar_path)
            A = np.sum(radar_path_data >= I_threshold)
            add_A = np.sum(radar_path_data >= I_threshold)
            A += add_A
            max_A = max_A if max_A>A else A 
        if max_A < A_threshold:
            count += 1
    print(f"{count} / {len(df_7km_dBZ)}") 

    
def median_filter(path: str, index_category: str):
    raw_data = np.load(path).astype(np.float32)
    
    # 归一化
    norm_params = {"dBZ": [0.0, 65.0], "ZDR": [-1.0, 5.0], "KDP":[-1.0, 6.0]}
    mmin, mmax = norm_params[index_category]
    raw_data[raw_data > mmax] = mmax
    raw_data[raw_data < mmin] = mmin
    raw_data = (raw_data - mmin)/(mmax - mmin)
    # 中值滤波
    img = ndimage.median_filter(raw_data, size=4)
    return np.array(img)

def mask_filter(path: str, index_category: str):
    raw_data = np.load(path).astype(np.float32)
    plt.imshow(raw_data, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f"raw_data-1.jpg")
    plt.close()
    var = np.var(raw_data)
    mean = np.mean(raw_data)
    mean_matrix = np.full((256, 256), mean)
    cov_matrix = np.cov(raw_data, rowvar=False)
    dm = np.sqrt(((np.transpose(raw_data-mean_matrix) @ cov_matrix)) @ ((raw_data - mean_matrix)**4))

    limit = mean + 3*var
    dm[dm > limit] = 0
    dm[dm >= 0] = 0
    dm[dm <= 71] = 0
    print(limit)
    plt.imshow(dm, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f"img-1.jpg")
    plt.close()
    

if __name__ == "__main__":
    # mask_filter("Dataset/NJU_CPOL_update2308/dBZ/1.0km/data_dir_000/frame_000.npy", index_category="dBZ")
    # kill_no_rain()
    print("Done!!!")