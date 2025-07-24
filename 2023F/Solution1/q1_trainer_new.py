import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
import torch.nn.functional as F

import sys
sys.path.append("2023F/Dataloader")
from data_loader_new import get_loader
sys.path.append("2023F/Solution1")
from feat_extractor import FeatureMatchDiscriminator
from feat_concator import FeatureMatchGenerator
import time

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    height = "3km"
    dataloader = get_loader(mode="train", height=height, question="1", batch_size=6)
    
    FE_Zh = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    FE_Zh = torch.nn.DataParallel(FE_Zh)
    optim_FE_Zh = torch.optim.AdamW(FE_Zh.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    FE_Zdr = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    FE_Zdr = torch.nn.DataParallel(FE_Zdr)
    optim_FE_Zdr = torch.optim.AdamW(FE_Zdr.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    FE_Kdp = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    FE_Kdp = torch.nn.DataParallel(FE_Kdp)
    optim_FE_Kdp = torch.optim.AdamW(FE_Kdp.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    FC_Zh = FeatureMatchGenerator().cuda()
    FC_Zh = torch.nn.DataParallel(FC_Zh)
    optim_FC_Zh = torch.optim.AdamW(FC_Zh.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)

    num_epochs = 50
    loss_list = []
    num_batchs = 500
    xx = np.arange(256)
    yy = np.arange(256)
    X, Y = np.meshgrid(xx, yy)
    cmap = matplotlib.cm.rainbow
    norm = matplotlib.colors.BoundaryNorm([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], cmap.N)
    
    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
        epoch_loss = 0.0
        for i, contents in enumerate(dataloader):                
            feats_Zh = FE_Zh(contents[f"Zh_{height}_10"].cuda())
            feats_Zdr = FE_Zdr(contents[f"Zdr_{height}_10"].cuda())
            feats_Kdp = FE_Kdp(contents[f"Kdp_{height}_10"].cuda())

            feats_total = torch.stack((feats_Zh[-2], feats_Zdr[-2], feats_Kdp[-2]), dim=2) # [B, 512, 3]
            feats_total = feats_total.unsqueeze(3)
            feats_total = feats_total.repeat(1, 1, 1, 3)
            
            
            real_Zh_1km = contents[f"Y_{height}_10"].cuda()
            pred_Zh_1km = FC_Zh(feats_total)
            
            plt.subplot(1, 2, 1)
            data = real_Zh_1km[0][4].cpu().detach().numpy()
            plt.contourf(X,Y, data, zdim='z',offset=0, cmap=cmap, norm=norm)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
            plt.subplot(1, 2, 2)
            data = pred_Zh_1km[0][4].cpu().detach().numpy()
            plt.contourf(X,Y, data, zdim='z',offset=0, cmap=cmap, norm=norm)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
            plt.savefig(f"trian_model.jpg")
            plt.close()

            # loss
            loss_mse = F.mse_loss(real_Zh_1km, pred_Zh_1km)
            epoch_loss += loss_mse
            optim_FE_Zh.zero_grad()
            optim_FE_Zdr.zero_grad()
            optim_FE_Kdp.zero_grad()
            optim_FC_Zh.zero_grad()
            
            loss_mse.backward()
            optim_FE_Zh.step()
            optim_FE_Zdr.step()
            optim_FE_Kdp.step()
            optim_FC_Zh.step()

            print(
                f"[Epoch {epoch}/{num_epochs}] [{i+1}/{len(dataloader)}] [{loss_mse.item()}]"
            )
        loss_list.append(epoch_loss.item() / num_batchs)
        if epoch % 10 == 0:
            torch.save(FE_Zh.state_dict(), f"2023F/Solution1/models/FE_new_{height}_Zh_{epoch}.pth")
            torch.save(FE_Kdp.state_dict(), f"2023F/Solution1/models/FE_new_{height}_Kdp_{epoch}.pth")
            torch.save(FE_Zdr.state_dict(), f"2023F/Solution1/models/FE_new_{height}_Zdr_{epoch}.pth")
            torch.save(FC_Zh.state_dict(), f"2023F/Solution1/models/FC_new_{height}_Zh_{epoch}.pth")
            df = pd.DataFrame({"MSE loss": loss_list})
            df.to_csv(f"2023F/Solution1/models/Train_MSE_loss_new_{height}.csv")
            
if __name__ == "__main__":
    train()