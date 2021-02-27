from mel_loader import get_data_loader
from model import ReferenceEncoder

import torch
from torch import nn
from torch.optim import Adam

from tqdm import tqdm

from tensorboardX import SummaryWriter

import numpy as np
import os
import json

if __name__ == '__main__':
    
    json_dict = json.load(open('config.json'))
    model_params_dict = json_dict['model_config']

    train_loader, valid_loader = get_data_loader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ReferenceEncoder(**model_params_dict)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    loss_func = nn.NLLLoss()

    train_writer = SummaryWriter()
    valid_writer = SummaryWriter(os.path.join(train_writer.logdir, 'eval'))

    step = 0

    for i in range(100):
        model.train()
        loss_list = list()
        acc_list = list()
        for batched_mel, batched_emo in tqdm(train_loader):
            optimizer.zero_grad()
            pred_tensor = model(batched_mel.to(device))
            loss = loss_func(pred_tensor, batched_emo.to(device))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            acc_list.append(torch.mean((torch.argmax(pred_tensor.cpu(), dim=-1) == batched_emo).float()))
            step += 1

            if step % 100 == 0:
                train_writer.add_scalar('loss', 
                                        np.mean(loss_list), 
                                        global_step=step)
                train_writer.add_scalar('acc', 
                                        np.mean(acc_list), 
                                        global_step=step)

        print(f'[Train] Loss: {np.mean(loss_list):2.3f} / Acc: {np.mean(acc_list):2.3f}')

        model.eval()
        loss_list = list()
        acc_list = list()
        for batched_mel, batched_emo in tqdm(valid_loader):

            pred_tensor = model(batched_mel.to(device))
            loss = loss_func(pred_tensor, batched_emo.to(device))

            loss_list.append(loss.item())
            acc_list.append(torch.mean((torch.argmax(pred_tensor.cpu(), dim=-1) == batched_emo).float()))
        # print(np.mean(loss_list))
        valid_writer.add_scalar('loss', 
                            np.mean(loss_list), 
                            global_step=step)
        valid_writer.add_scalar('acc', 
                            np.mean(acc_list), 
                            global_step=step)

        print(f'[Eval] Loss: {np.mean(loss_list):2.3f} / Acc: {np.mean(acc_list):2.3f}')
