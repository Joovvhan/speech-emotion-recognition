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

from train import save_model, load_model

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

    # train_loader, valid_loader = get_data_loader(batch_size)
    _, valid_loader = get_data_loader(batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'GST-RNN':
        model = ReferenceEncoder(**model_params_dict)
    elif args.model == 'GST-RNN-Mod':
        model = ReferenceEncoderMod(**model_params_dict)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    loss_func = nn.NLLLoss()

    step = 0

    assert args.run is not None, f'{args.run} is not a valid run name'

    run_path = os.path.join('runs', args.run)
    assert os.path.isdir(run_path), f'Invalid path {run_path}'

    model.eval()

    loss_list = list()
    acc_list = list()

    infos = list()

    hs = torch.rand([0, 128])
    accs = torch.rand([0])

    for batched_mel, batched_emo, info in tqdm(valid_loader):
        
        # 'ANAD/1sec_segmented_part3/1sec_segmented_part3/V7_1 (140).wav',
        # 'ANAD',
        # 'anger',

        h, pred_tensor = model.infer(batched_mel.to(device))
        h = h.detach().cpu()
        hs = torch.cat((hs, h), axis=0)

        # acc_list.append(torch.mean((torch.argmax(pred_tensor.cpu(), dim=-1) == batched_emo).float()))

        acc = (torch.argmax(pred_tensor.cpu(), dim=-1) == batched_emo).float()
        accs = torch.cat((accs, acc), axis=0)

        infos.append(info)

    np.savez('infer.npz', hs=hs, accs=accs, infos=infos)