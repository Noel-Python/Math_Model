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
from data_loader_rain import get_loader
sys.path.append("2023F/Solution2")
from ResUnet import Encoder, Decoder
import time

def trainer():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    height = "3km"
    dataloader = get_loader(mode="train", height=height, question="1", batch_size=16)
    
    FE_Zh = Encoder(10).cuda()
    optim_FE_Zh = torch.optim.AdamW(FE_Zh.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    FE_Zdr = Encoder(10).cuda()
    optim_FE_Zdr = torch.optim.AdamW(FE_Zdr.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    FC = Decoder(10, 2).cuda()
    optim_FC = torch.optim.AdamW(FC.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=0.03)
    
    num_epochs = 50
    loss_list = []
    xx = np.arange(256)
    yy = np.arange(256)
    X, Y = np.meshgrid(xx, yy)
    cmap = matplotlib.cm.rainbow
    norm = matplotlib.colors.BoundaryNorm([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], cmap.N)
    
    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
        epoch_loss = 0.0
        for i, contents in enumerate(dataloader):
            feats_Zh = FE_Zh(contents[f"Zh_{height}_10"].cuda())
            feats_Zdr = FE_Zdr(contents[f"Zdr_{height}_10"].cuda())

            feats_total = [torch.cat((feats_Zh[i], feats_Zdr[i]), dim=1) for i in range(8)]

            output = FC(feats_total)
            result = output[-1]
            
            real_Zh_1km = contents[f"Y_10"].cuda()
            pred_Zh_1km = result
            
            # loss
            loss_mse = F.mse_loss(real_Zh_1km, pred_Zh_1km)
            epoch_loss += loss_mse
            
            optim_FE_Zh.zero_grad()
            optim_FE_Zdr.zero_grad()
            optim_FC.zero_grad()
            loss_mse.backward()
            optim_FE_Zh.step()
            optim_FE_Zdr.step()
            optim_FC.step()
            
            plt.subplot(1, 2, 1)
            data = real_Zh_1km[0][0].cpu().detach().numpy()
            plt.contourf(X,Y, data, zdim='z',offset=0, cmap=cmap, norm=norm)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
            plt.subplot(1, 2, 2)
            data = pred_Zh_1km[0][0].cpu().detach().numpy()
            plt.contourf(X,Y, data, zdim='z',offset=0, cmap=cmap, norm=norm)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
            plt.savefig(f"trian_model.jpg")
            plt.close()
            
            print(
                f"[Epoch {epoch}/{num_epochs}] [{i+1}/{len(dataloader)}] [{loss_mse.item()}]"
            )
        loss_list.append(epoch_loss.item() / len(dataloader))
        if epoch % 10 == 0:
            torch.save(FE_Zh.state_dict(), f"2023F/Solution4/zhzdr/FE_{height}_Zh_{epoch}.pth")
            torch.save(FE_Zdr.state_dict(), f"2023F/Solution4/zhzdr/FE_{height}_Zdr_{epoch}.pth")
            torch.save(FC.state_dict(), f"2023F/Solution4/zhzdr/FC_{height}_rain_{epoch}.pth")
            df = pd.DataFrame({"MSE loss": loss_list})
            df.to_csv(f"2023F/Solution4/zhzdr/Train_MSE_loss_{height}.csv")
if __name__ == "__main__":
    trainer()