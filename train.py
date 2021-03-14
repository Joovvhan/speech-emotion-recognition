from mel_loader import get_data_loader
from model import ReferenceEncoder, ReferenceEncoderMod

import argparse

import torch
from torch import nn
from torch.optim import Adam

from tqdm import tqdm

from tensorboardX import SummaryWriter

import numpy as np
import os
import json

def save_model(model, optim, path, steps):
    
    save_path = os.path.join(path, 'checkpoint.pt')
    
    torch.save({
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            }, save_path)

    return save_path

def load_model(model, optim, path):
    
    load_path = os.path.join(path, 'checkpoint.pt')
    checkpoint = torch.load(load_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
            
    return checkpoint['steps']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name', default='GST-RNN')
    parser.add_argument('--run', type=str, help='run name', default=None)
    parser.add_argument('--log_step', type=int, help='logging step', default=100)
    parser.add_argument('--save_step', type=int, help='saving step', default=1000)
    parser.add_argument('--num_iter', type=int, help='number of iteration', default=100)
    args = parser.parse_args()
    
    json_dict = json.load(open('config.json'))
    model_params_dict = json_dict['model_config']

    batch_size = 32

    train_loader, valid_loader = get_data_loader(batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'GST-RNN':
        model = ReferenceEncoder(**model_params_dict)
    elif args.model == 'GST-RNN-Mod':
        model = ReferenceEncoderMod(**model_params_dict)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    loss_func = nn.NLLLoss()

    step = 0

    if args.run is None:
        train_writer = SummaryWriter()
        valid_writer = SummaryWriter(os.path.join(train_writer.logdir, 'eval'))

    else:
        run_path = os.path.join('runs', args.run)
        assert os.path.isdir(run_path), f'Invalid path {run_path}'
        train_writer = SummaryWriter(run_path)
        valid_writer = SummaryWriter(os.path.join(train_writer.logdir, 'eval'))
        step = load_model(model, optimizer, train_writer.logdir)


    for i in range(args.num_iter):
        model.train()
        loss_list = list()
        acc_list = list()
        for batched_mel, batched_emo, _ in tqdm(train_loader):
            optimizer.zero_grad()
            pred_tensor = model(batched_mel.to(device))
            loss = loss_func(pred_tensor, batched_emo.to(device))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            acc_list.append(torch.mean((torch.argmax(pred_tensor.cpu(), dim=-1) == batched_emo).float()))
            step += 1

            if step % args.log_step == 0:
                train_writer.add_scalar('loss', 
                                        np.mean(loss_list), 
                                        global_step=step)
                train_writer.add_scalar('acc', 
                                        np.mean(acc_list), 
                                        global_step=step)
                loss_list = list()
                acc_list = list()

            if step % args.save_step == 0:
                save_model(model, optimizer, train_writer.logdir, step)
                pass

        print(f'[Train] Loss: {np.mean(loss_list):2.3f} / Acc: {np.mean(acc_list):2.3f}')

        model.eval()
        loss_list = list()
        acc_list = list()
        for batched_mel, batched_emo, _ in tqdm(valid_loader):

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
