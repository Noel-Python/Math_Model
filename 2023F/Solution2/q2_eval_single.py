import os
import torch
import tqdm
import time
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append("2023F/Dataloader")
from data_loader_new import get_loader

sys.path.append("2023F/Solution2")
from ResUnet import Encoder, Decoder

sys.path.append("2023F/Utils")
from scores import score_cavr, score_csi, score_far, score_mse, score_pod, score_ssim

def eval_single():
    # random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    height = "1km"
    threshold_abs = 35
    threshold = threshold_abs / 65.0
    frame_id = 20
    
    l = 0.92
    b = 0.12
    w = 0.015
    h = 1-2*b
    rect = [l, b, w, h]
    
    dataloader = get_loader(mode="test", height=height, question="1", batch_size=1, drop_last=False)
    
    weights = torch.load(f"2023F/Solution2/models/FE_new_{height}_Zh_50.pth", map_location=torch.device('cpu'))
    FE_Zh = Encoder(10).cuda()
    # FE_Zh = torch.nn.DataParallel(FE_Zh)
    # FE_Zh.load_state_dict(weights['model_state_dict'])
    FE_Zh.load_state_dict(weights)

    weights = torch.load(f"2023F/Solution2/models/FE_new_{height}_Zdr_50.pth", map_location=torch.device('cpu'))
    FE_Zdr = Encoder(10).cuda()
    # FE_Zdr = torch.nn.DataParallel(FE_Zdr)
    # FE_Zdr.load_state_dict(weights['model_state_dict'])
    FE_Zdr.load_state_dict(weights)

    weights = torch.load(f"2023F/Solution2/models/FE_new_{height}_Kdp_50.pth", map_location=torch.device('cpu'))
    FE_Kdp = Encoder(10).cuda()
    # FE_Kdp = torch.nn.DataParallel(FE_Kdp)
    # FE_Kdp.load_state_dict(weights['model_state_dict'])
    FE_Kdp.load_state_dict(weights)
    
    weights = torch.load(f"2023F/Solution2/models/FC_new_{height}_Zh_50.pth", map_location=torch.device('cpu'))
    FC = Decoder(10, 3).cuda()
    # FC = torch.nn.DataParallel(FC)
    # FC.load_state_dict(weights['model_state_dict'])
    FC.load_state_dict(weights)

    xx = np.arange(256)
    yy = np.arange(256)
    X, Y = np.meshgrid(xx, yy)
    cmap = matplotlib.cm.rainbow
    norm = matplotlib.colors.BoundaryNorm([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], cmap.N)
    
    for i, contents in enumerate(dataloader):
        if i == frame_id:
            FE_Zh.eval()
            FE_Zdr.eval()
            FE_Kdp.eval()
            FC.eval()
            feats_Zh = FE_Zh(contents[f"Zh_{height}_10"].cuda())
            feats_Zdr = FE_Zdr(contents[f"Zdr_{height}_10"].cuda())
            feats_Kdp = FE_Kdp(contents[f"Kdp_{height}_10"].cuda())
            
            feats_total = [torch.cat((feats_Zh[i], feats_Zdr[i], feats_Kdp[i]), dim=1) for i in range(8)]
            output = FC(feats_total)
            result = output[-1]
            
            real_Zh = contents[f"Y_{height}_10"].detach().numpy()
            pred_Zh = result.cpu().detach().numpy()

            mse_scores = []
            ssim_scores = []
            pod_scores = []
            far_scores = []
            csi_scores = []
            
            fig = plt.figure(figsize=(21, 16), dpi=400)
            
            for batch in range(real_Zh.shape[0]):
                channel_mse = []
                channel_ssim = []
                channel_pod = []
                channel_far = []
                channel_csi = []
                for channel in range(real_Zh.shape[1]):
                    real = real_Zh[batch][channel]
                    pred = pred_Zh[batch][channel]
                    
                    # All Scores
                    channel_mse.append(score_mse(real, pred))
                    channel_ssim.append(score_ssim(real, pred))
                    channel_pod.append(score_pod(real, pred, threshold=threshold))
                    channel_far.append(score_far(real, pred, threshold=threshold))
                    channel_csi.append(score_csi(real, pred, threshold=threshold))
                    
                    # PLot
                    ax1 = fig.add_subplot(4, 5, int(channel + 1 + (int(channel) // 5)*5))
                    ax1.contourf(X,Y, real, zdim='z',offset=0, cmap=cmap, norm=norm)
                    ax1.set_title(f"Real Zh : t+{channel+1}")    
                    
                    ax2 = fig.add_subplot(4, 5, int(channel + 1 +5 + (int(channel) // 5)*5))
                    ax2.contourf(X,Y, pred, zdim='z',offset=0, cmap=cmap, norm=norm)
                    ax2.set_title(f"Pred Zh : t+{channel+1}") 
                    
                mse_scores.append(channel_mse)
                ssim_scores.append(channel_ssim)
                pod_scores.append(channel_pod)
                far_scores.append(channel_far)
                csi_scores.append(channel_csi)
                
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes(rect)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            plt.savefig(f"2023F/Solution2/results/{frame_id}_{height}_{threshold_abs}.jpg")
            plt.close() 
            
            # batch_mean
            mse_scores = np.mean(np.array(mse_scores), axis=0)
            ssim_scores = np.mean(np.array(ssim_scores), axis=0)
            pod_scores = np.mean(np.array(pod_scores), axis=0)
            far_scores = np.mean(np.array(far_scores), axis=0)
            csi_scores = np.mean(np.array(csi_scores), axis=0)
            
            all_scores = pd.DataFrame({"mse": mse_scores, "ssim": ssim_scores, "pod": pod_scores, "far": far_scores, "csi": csi_scores})
            all_scores.to_csv(f"2023F/Solution2/results/{frame_id}_{height}_{threshold_abs}.csv")
            print("Done")
        
if __name__ == "__main__":
    eval_single()