import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import torchaudio
import os
import math
import random
import sys
from model2_scaling import *
from tensorboardX import SummaryWriter
from mydataset_magnitude import *
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--exp_day', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float,required=True)
    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp_day = args.exp_day
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    start_epoch = 0
    num_epochs = 100

    random.seed(0) # fixing seed for fixed validation set

    # prepare train data
    train_label_path = '' # directory of txt file containing all data pair directory
    train_list = prepare_data(train_label_path)
    random.shuffle(train_list)   
    num_valid = int(0.1*len(train_list))
    valid_list = train_list[len(train_list)-num_valid:len(train_list)]
    train_list = train_list[0:len(train_list)-num_valid]
    print(len(train_list))
    print(len(valid_list))

    # define Dataset & Dataloader
    train_dataset = Mydataset2(train_list)
    valid_dataset = Mydataset2(valid_list)
    print("dataset done!")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=lambda x: my_collate(x))
    val_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=lambda x: my_collate(x))
    
    tensorboard_path = 'runs/model1/'+str(exp_day)+'/learning_rate_'+str(learning_rate)+'_batch_'+str(batch_size)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    summary = SummaryWriter(tensorboard_path)

    # create model, criterion, optimizer, scheduler instance 
    model = QNet().to(device)
    print(count_parameters(model))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=2,min_lr=0.000005)
    
    # train
    best_loss = 10
    for epoch in range(start_epoch,num_epochs):
        model.train()
        train_loss=0
        for i, (m_stft,PESQ) in enumerate(tqdm(train_loader)):
            data = m_stft.to(device)
            PESQ = PESQ.to(device)
            PESQ = PESQ.unsqueeze(1)
            outputs = model(data)
            loss = criterion(outputs, PESQ).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        train_loss = train_loss/len(train_loader)
        summary.add_scalar('training loss',train_loss,epoch)

        modelsave_path = 'model_ckpt/model1/'+str(exp_day)+'/'+'learning_rate_'+str(learning_rate)+'_batch_'+str(batch_size)
        if not os.path.exists(modelsave_path):
            os.makedirs(modelsave_path)
        torch.save(model.state_dict(), str(modelsave_path)+'/epoch'+str(epoch)+'model.pth')
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pth')
        
        # validation during training
        model.eval()
        with torch.no_grad():
            val_loss =0.
            for j, (m_stft,PESQ) in enumerate(tqdm(val_loader)):
                
                data = m_stft.to(device)
                PESQ = PESQ.to(device)
                PESQ = PESQ.unsqueeze(1)
                outputs = model(data)

                loss = criterion(outputs, PESQ).to(device)
                val_loss +=loss.item()

            val_loss = val_loss/len(val_loader)
            summary.add_scalar('val loss',val_loss,epoch)
            summary.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
            scheduler.step(val_loss)
            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pth')
                best_loss = val_loss
               
               
             
