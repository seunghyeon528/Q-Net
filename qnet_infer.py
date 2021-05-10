import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
# import torchaudio
import os
import math
import random
import sys
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import pdb

from model2_scaling import *
from mydataset_magnitude import *

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import scipy.stats
if __name__ == '__main__':
    
    # device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # hyperparameter
    batch_size = 50

    # Load Bestmodel (Q-Net, Discriminator)
    model = QNet().to(device)
    bestmodel_path = './lastmodel.pth'
    model.load_state_dict(torch.load(bestmodel_path,map_location = device))
    
    # prepare test data
    random.seed(0)    
    train_label_path = '/home/nas/DB/[DB]_Qnet/train_all_0216.txt'
    train_list = prepare_data(train_label_path)
    random.shuffle(train_list)   
    num_valid = int(0.5*len(train_list))
    valid_list = train_list[len(train_list)-num_valid:len(train_list)]
    print(len(train_list))
	
    # make valid
    test_dataset = Mydataset2(valid_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=lambda x: my_collate(x))
    train_dataset = Mydataset2(train_list)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=lambda x: my_collate(x))
    true_PESQ_list = torch.Tensor([]).to(device)
    infer_PESQ_list = torch.Tensor([]).to(device)

    # infer
    model.eval()
    with torch.no_grad():
        for j, (m_stft,PESQ) in enumerate(tqdm(test_loader)):
            
            data = m_stft.to(device)
            true_PESQ = PESQ.to(device)
            true_PESQ_list = torch.cat([true_PESQ_list, true_PESQ], dim=0)
            true_PESQ = true_PESQ.unsqueeze(1)    
            outputs = model(data)
            infer_PESQ = outputs.squeeze(1)
            infer_PESQ_list = torch.cat([infer_PESQ_list,infer_PESQ],dim=0)

                
    # Result Type conversion (tensor -> numpy)
    true_PESQ_list = true_PESQ_list.cpu() 
    true_PESQ_list = true_PESQ_list.numpy()
    infer_PESQ_list = infer_PESQ_list.cpu()
    infer_PESQ_list = infer_PESQ_list.numpy()

    # Plot two graphs (one is y = x, the other is scatter graph of inferred data)
    plt.scatter(true_PESQ_list,infer_PESQ_list,color = 'c')
    plt.scatter(true_PESQ_list,true_PESQ_list,color = 'magenta')
    title = bestmodel_path.split('/')[-3]
    plt.title(title,fontsize = 20)
    plt.xlabel('True', fontsize = 16)
    plt.ylabel('Infer',fontsize = 16)
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    save_path = './'+ title +'plot.png'
    # plt.savefig(save_path)
    plt.savefig('./plot.png')
    plt.show()   
	
    
    print(scipy.stats.spearmanr(true_PESQ_list,infer_PESQ_list))
    print(scipy.stats.pearsonr(true_PESQ_list,infer_PESQ_list))   
