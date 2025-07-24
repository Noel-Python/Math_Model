import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
import warnings
warnings.filterwarnings("ignore")

dir_list = ["zh/", "zhzdr/", "zhkdp/", "zdrkdp/", "zhzdrkdp/"]
root_path = "2023F/Solution4/"


def plot_all():
    df_zh = pd.read_csv(root_path+dir_list[0]+"3km_35_zh.csv")
    df_zhzdr = pd.read_csv(root_path+dir_list[1]+"3km_35_zhzdr.csv")
    df_zhkdp = pd.read_csv(root_path+dir_list[2]+"3km_35_zhkdp.csv")
    df_zdrkdp = pd.read_csv(root_path+dir_list[3]+"3km_35_zdrkdp.csv")
    df_zhzdrkdp = pd.read_csv(root_path+dir_list[4]+"3km_35_zhzdrkdp.csv")
    label_list = ["Zh", "Zh & Zdr", "Zh & Kdp", "Zdr & Kdp", "Zh & Zdr & Kdp"]
    marker_list = ["s", "^", ".", "*", "p"]
    
    for score in ["mse", "ssim", "pod", "far", "csi"]:
        fig, ax = plt.subplots(1, 1)
        for i, df in enumerate([df_zh, df_zhzdrkdp,  df_zhkdp, df_zdrkdp, df_zhzdr]):
            ax.plot(df[score].tolist(), marker=marker_list[i], label=label_list[i])
        plt.title(f"{score}逐帧变化", fontproperties = zhfont)
        plt.xlabel("预测时间/帧数",  fontproperties = zhfont)
        plt.ylabel(f"{score}", fontproperties = zhfont)
        plt.legend()
        plt.savefig(f"2023F/Solution4/results/all_{score}.png")
        plt.close()    
        
    
def plot_single():
    df_zh = pd.read_csv(root_path+dir_list[0]+"20_3km_35_zh.csv")
    df_zhzdr = pd.read_csv(root_path+dir_list[1]+"20_3km_35_zhzdr.csv")
    df_zhkdp = pd.read_csv(root_path+dir_list[2]+"20_3km_35_zhkdp.csv")
    df_zdrkdp = pd.read_csv(root_path+dir_list[3]+"20_3km_35_zdrkdp.csv")
    df_zhzdrkdp = pd.read_csv(root_path+dir_list[4]+"20_3km_35_zhzdrkdp.csv")
    label_list = ["Zh", "Zh & Zdr", "Zh & Kdp", "Zdr & Kdp", "Zh & Zdr & Kdp"]
    marker_list = ["s", "^", ".", "*", "p"]
    
    for score in ["mse", "ssim", "pod", "far", "csi"]:
        fig, ax = plt.subplots(1, 1)
        for i, df in enumerate([df_zh, df_zhzdrkdp,  df_zhkdp, df_zdrkdp, df_zhzdr]):
            ax.plot(df[score].tolist(), marker=marker_list[i], label=label_list[i])
        plt.title(f"{score}逐帧变化", fontproperties = zhfont)
        plt.xlabel("预测时间/帧数",  fontproperties = zhfont)
        plt.ylabel(f"{score}", fontproperties = zhfont)
        plt.legend()
        plt.savefig(f"2023F/Solution4/results/single_{score}.png")
        plt.close()  
    
if __name__ == "__main__":
    plot_all()
    plot_single()
    print("Done!!!")