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
from data_loader import get_loader
sys.path.append("2023F/Solution1")
from feat_extractor import FeatureMatchDiscriminator
from feat_concator import FeatureMatchGenerator

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    dataloader = get_loader(mode="train", question="1", batch_size=6)
    
    FE_1km_Zh = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    FE_1km_Zh = torch.nn.DataParallel(FE_1km_Zh)
    #FE_3km_Zh = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    #FE_7km_Zh = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    optim_FE_1km_Zh = torch.optim.Adam(FE_1km_Zh.parameters(), lr=1e-3, betas=[0.0, 0.9])
    #optim_FE_3km_Zh = torch.optim.Adam(FE_3km_Zh.parameters(), lr=1e-4, betas=[0.0, 0.9])
    #optim_FE_7km_Zh = torch.optim.Adam(FE_7km_Zh.parameters(), lr=1e-4, betas=[0.0, 0.9])
    
    FE_1km_Zdr = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    FE_1km_Zdr = torch.nn.DataParallel(FE_1km_Zdr)
    #FE_3km_Zdr = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    #FE_7km_Zdr = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    optim_FE_1km_Zdr = torch.optim.Adam(FE_1km_Zdr.parameters(), lr=1e-3, betas=[0.0, 0.9])
    #optim_FE_3km_Zdr = torch.optim.Adam(FE_3km_Zdr.parameters(), lr=1e-4, betas=[0.0, 0.9])
    #optim_FE_7km_Zdr = torch.optim.Adam(FE_7km_Zdr.parameters(), lr=1e-4, betas=[0.0, 0.9])
    
    FE_1km_Kdp = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    FE_1km_Kdp = torch.nn.DataParallel(FE_1km_Kdp)
    #FE_3km_Kdp = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    #FE_7km_Kdp = FeatureMatchDiscriminator(img_size=256, conv_dim=64).cuda()
    optim_FE_1km_Kdp = torch.optim.Adam(FE_1km_Kdp.parameters(), lr=1e-3, betas=[0.0, 0.9])
    #optim_FE_3km_Kdp = torch.optim.Adam(FE_3km_Kdp.parameters(), lr=1e-4, betas=[0.0, 0.9])
    #optim_FE_7km_Kdp = torch.optim.Adam(FE_7km_Kdp.parameters(), lr=1e-4, betas=[0.0, 0.9])
    
    FC_1km_Zh = FeatureMatchGenerator().cuda()
    FC_1km_Zh = torch.nn.DataParallel(FC_1km_Zh)
    #FC_3km_Zh = FeatureMatchGenerator().cuda()
    #FC_7km_Zh = FeatureMatchGenerator().cuda()
    optim_FC_1km_Zh = torch.optim.Adam(FC_1km_Zh.parameters(), lr=1e-3, betas=[0.0, 0.9])
    #optim_FC_3km_Zh = torch.optim.Adam(FC_3km_Zh.parameters(), lr=1e-4, betas=[0.0, 0.9])
    #optim_FC_7km_Zh = torch.optim.Adam(FC_7km_Zh.parameters(), lr=1e-4, betas=[0.0, 0.9])
    
    
    num_epochs = 20
    loss_list = []
    num_batchs = 500
    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
        epoch_loss = 0.0
        for i, contents in enumerate(dataloader):
            if i > num_batchs:
                break                   
            feats_1km_Zh = FE_1km_Zh(contents["Zh_1km_10"].cuda())
            #feats_3km_Zh = FE_3km_Zh(contents["Zh_3km_10"].cuda())
            #feats_7km_Zh = FE_7km_Zh(contents["Zh_7km_10"].cuda())
            
            feats_1km_Zdr = FE_1km_Zdr(contents["Zdr_1km_10"].cuda())
            #feats_3km_Zdr = FE_3km_Zdr(contents["Zdr_3km_10"].cuda())
            #feats_7km_Zdr = FE_7km_Zdr(contents["Zdr_7km_10"].cuda())
            
            feats_1km_Kdp = FE_1km_Kdp(contents["Kdp_1km_10"].cuda()) #[B, 512]
            #feats_3km_Kdp = FE_3km_Kdp(contents["Kdp_3km_10"].cuda())
            #feats_7km_Kdp = FE_7km_Kdp(contents["Kdp_7km_10"].cuda()) 
            
            # feats_Zh = torch.stack((feats_1km_Zh[-2], feats_3km_Zh[-2], feats_7km_Zh[-2]), dim=2)
            # feats_Zdr = torch.stack((feats_1km_Zdr[-2], feats_3km_Zdr[-2], feats_7km_Zdr[-2]), dim=2)
            # feats_Kdp = torch.stack((feats_1km_Kdp[-2], feats_3km_Kdp[-2], feats_7km_Kdp[-2]), dim=2)
            
            # feats_total = torch.stack((feats_Zh, feats_Zdr, feats_Kdp), dim=3)
            feats_total = torch.stack((feats_1km_Zh[-2], feats_1km_Zdr[-2], feats_1km_Kdp[-2]), dim=2) # [B, 512, 3]
            feats_total = feats_total.unsqueeze(3)
            feats_total = feats_total.repeat(1, 1, 1, 3)
            
            
            real_Zh_1km = contents["Y_1km_10"].cuda()
            #real_Zh_3km = contents["Y_3km_10"].cuda()
            #real_Zh_7km = contents["Y_7km_10"].cuda()
            pred_Zh_1km = FC_1km_Zh(feats_total)
            
            plt.subplot(1, 2, 1)
            norm = matplotlib.colors.Normalize(vmin=0.0,vmax=1.0)
            plt.imshow(real_Zh_1km[0][4].cpu().detach().numpy(), cmap='coolwarm', interpolation='nearest', norm=norm)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(pred_Zh_1km[0][4].cpu().detach().numpy(), cmap='coolwarm', interpolation='nearest', norm=norm)
            plt.colorbar()
            plt.savefig(f"trian_model.jpg")
            plt.close()
            #pred_Zh_3km = FC_3km_Zh(feats_total)
            #pred_Zh_7km = FC_7km_Zh(feats_total)
            
            
            # loss
            loss_mse = F.mse_loss(real_Zh_1km, pred_Zh_1km) * 100.0 #+ F.mse_loss(real_Zh_3km, real_Zh_3km) + F.mse_loss(real_Zh_7km, real_Zh_7km)
            epoch_loss += loss_mse
            optim_FE_1km_Zh.zero_grad()
            #optim_FE_3km_Zh.zero_grad()
            #optim_FE_7km_Zh.zero_grad()
            optim_FE_1km_Zdr.zero_grad()
            #optim_FE_3km_Zdr.zero_grad()
            #optim_FE_7km_Zdr.zero_grad()
            optim_FE_1km_Kdp.zero_grad()
            #optim_FE_3km_Kdp.zero_grad()
            #optim_FE_7km_Kdp.zero_grad()
            optim_FC_1km_Zh.zero_grad()
            #optim_FC_3km_Zh.zero_grad()
            #optim_FC_7km_Zh.zero_grad()
            
            loss_mse.backward()
            optim_FE_1km_Zh.step()
            #optim_FE_3km_Zh.step()
            #optim_FE_7km_Zh.step()
            optim_FE_1km_Zdr.step()
            #optim_FE_3km_Zdr.step()
            #optim_FE_7km_Zdr.step()
            optim_FE_1km_Kdp.step()
            #optim_FE_3km_Kdp.step()
            #optim_FE_7km_Kdp.step()
            optim_FC_1km_Zh.step()
            #optim_FC_3km_Zh.step()
            #optim_FC_7km_Zh.step()

            print(
                f"[Epoch {epoch}/{num_epochs}] [{i}/{len(dataloader)}] [{loss_mse.item()}]"
            )
        loss_list.append(epoch_loss.item() / num_batchs)
        if epoch % 5 == 0:
            torch.save({"model_state_dict": FE_1km_Zh.state_dict()}, f"2023F/Solution1/models/FE_1km_Zh_{epoch}.pth")
            torch.save({"model_state_dict": FE_1km_Kdp.state_dict()}, f"2023F/Solution1/models/FE_1km_Kdp_{epoch}.pth")
            torch.save({"model_state_dict": FE_1km_Zdr.state_dict()}, f"2023F/Solution1/models/FE_1km_Zdr_{epoch}.pth")
            torch.save({"model_state_dict": FC_1km_Zh.state_dict()}, f"2023F/Solution1/models/FC_1km_Zh_{epoch}.pth")
            df = pd.DataFrame({"MSE loss": loss_list})
            df.to_csv("Train_MSE_loss")
                 
    
if __name__ == "__main__":
    train()