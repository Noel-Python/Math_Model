import numpy as np
import pandas as pd
import os

#读取降水量数据，保存为csv结构化数据
def read_rainfall_path():
    rainfall_root_path = "Dataset/NJU_CPOL_kdpRain"
    rain_id_series = []
    frame_id_series = []
    rainfall_path_series = []
    rain_id_list = os.listdir(rainfall_root_path)
    for rain_id in rain_id_list:
        rain_id_path = os.path.join(rainfall_root_path, rain_id)
        frame_id_list = os.listdir(rain_id_path)
        for frame_id in frame_id_list:
            frame_id_path = os.path.join(rain_id_path, frame_id)
            rain_id_series.append(rain_id)
            frame_id_series.append(frame_id)
            rainfall_path_series.append(frame_id_path)
    
    df_rainfall = pd.DataFrame({"rain_id": rain_id_series, "frame_id": frame_id_series, "rainfall_path": rainfall_path_series})
    df_rainfall = df_rainfall.sort_values(by=['rain_id', 'frame_id'], ascending=[True, True], ignore_index=True)
    df_rainfall.to_csv("Dataset/rainfall_path.csv") 

#读取雷达信号数据，保存为csv结构化数据
def read_radar_path():
    radar_root_path = "Dataset/NJU_CPOL_update2308"
    index_category_list = os.listdir(radar_root_path)
    index_category_series = []
    height_series = []
    rain_id_series = []
    frame_id_series = [] 
    radar_path_series = []
    for index_category in index_category_list:
        index_category_path = os.path.join(radar_root_path, index_category)
        height_list = os.listdir(index_category_path)
        for height in height_list:
            height_path = os.path.join(index_category_path, height)
            rain_id_list = os.listdir(height_path)
            for rain_id in rain_id_list:
                rain_id_path = os.path.join(height_path, rain_id)
                frame_id_list = os.listdir(rain_id_path)
                for frame_id in frame_id_list:
                    frame_id_path = os.path.join(rain_id_path, frame_id)
                    index_category_series.append(index_category)
                    height_series.append(height)
                    rain_id_series.append(rain_id)
                    frame_id_series.append(frame_id)
                    radar_path_series.append(frame_id_path)
    
    df_radar = pd.DataFrame({"rain_id": rain_id_series, "frame_id": frame_id_series, "height": height_series, "index_category": index_category_series, "radar_path": radar_path_series})
    df_radar = df_radar.sort_values(by=["rain_id", "index_category", "height", "frame_id"], ascending=[True, True, True, True], ignore_index=True)
    df_radar.to_csv("Dataset/radar_path.csv") 

# 划分数据集
def partition_dataset(time_step=10):
    df = pd.read_csv("Dataset/radar_path.csv")
    rain_id_unique_list = df["rain_id"].unique().tolist()
    height_unique_list = df["height"].unique().tolist()
    index_category_list = df["index_category"].unique().tolist()
    
    rain_id_series = []
    height_series = []
    index_category_series = []
    input_series = []
    output_series = []
    
    for rain_id_unique in rain_id_unique_list:
        df_rain_id = df[df["rain_id"] == rain_id_unique]
        for height in height_unique_list:
            df_rain_id_height = df_rain_id[df_rain_id["height"] == height]
            for index_category in index_category_list:
                df_rain_id_height_index = df_rain_id_height[df_rain_id_height["index_category"] == index_category]
                radar_path_list = df_rain_id_height_index["radar_path"].tolist()
                
                num = int(len(radar_path_list) - 2 * time_step + 1)
                assert num>0
                for i in range(num):
                    input_list = []
                    output_list = []
                    for j in range(10):
                        input_list.append(radar_path_list[i+j])
                        output_list.append(radar_path_list[i+j+10])
                    rain_id_series.append(rain_id_unique)
                    height_series.append(height)
                    index_category_series.append(index_category)
                    input_series.append(input_list)
                    output_series.append(output_list)
                
    df_radar = pd.DataFrame({"rain_id": rain_id_series, "height": height_series, "index_category": index_category_series, "input": input_series, "output": output_series})
    df_radar.to_csv("Dataset/radar_raw_dataset.csv")

# 将数据集划分训练集和测试集
def dataset_divid():
    df_3km_kill_norain = pd.read_csv("Dataset/radar_3km_kill_norain.csv")
    df_3km_kill_norain = df_3km_kill_norain.reindex()
    sample_size = int(len(df_3km_kill_norain) * 0.9)
    df_3km_kill_norain_train = df_3km_kill_norain.sample(sample_size)
    df_3km_kill_norain_test = df_3km_kill_norain.drop(df_3km_kill_norain_train.index)
    df_3km_kill_norain_train.to_csv("Dataset/df_3km_kill_norain_train.csv")
    df_3km_kill_norain_test.to_csv("Dataset/df_3km_kill_norain_test.csv")

if __name__ == "__main__":
    # read_rainfall_path()
    # read_radar_path()
    # partition_dataset()
    dataset_divid()
    print("Done!!!")
    