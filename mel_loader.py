import sys
from glob import glob
import json
sys.path.append('tacotron2')
# print(sys.path)
import os
import random
from functools import partial

import itertools

import librosa
import numpy as np

from types import SimpleNamespace

from tacotron2.data_utils import TextMelLoader
from tacotron2 import layers

from meta_loader import get_meta, emotions
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class CustomMelLoader(TextMelLoader):
    
    def __init__(self, hparams):
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        
    def get_mel(self, audio, sampling_rate):
        if not self.load_mel_from_disk:
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = torch.tensor(audio / self.max_wav_value, dtype=torch.float).unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

hparams = json.load(open('hparams.json', 'r'), object_hook=lambda x: SimpleNamespace(**x))
MEL_LOADER = CustomMelLoader(hparams)
    
def load_wav(file, save_resampled=False):
    
    def convert_to_resampled_file(file):
        resampled_file = os.path.join('resampled', file).replace('.raw', '.wav')
        return resampled_file

    resampled_file = convert_to_resampled_file(file)
    
    if False and os.path.isfile(resampled_file):
        y, fs = librosa.core.load(file, sr=None)
    else:
        if '.raw' in file:
            y = np.fromfile(file, dtype=np.int16)
#             y = y / 2 ** 15

        elif '.wav' in file:

            y, fs = librosa.core.load(file, sr=None)

            if fs == 16000:
                y = (y * 2 ** 15).astype(np.int16)
                # return y
                pass
            elif fs % 16000 == 0:
                step = int(fs / 16000)
                offset = random.randint(0, step)
                # return y[offset::step] * 2 ** 15
                y = (y[offset::step] * 2 ** 15).astype(np.int16)
                pass
            else:
                y, fs = librosa.core.load(file, sr=16000)
                y = (y * 2 ** 15).astype(np.int16)
        else: assert False, f'Invalid File Format {file}'
        
        if save_resampled:
            os.makedirs(os.path.dirname(resampled_file), exist_ok=True)
#             int_y = (y * 2 ** 15).astype(np.int16)
            wavfile.write(resampled_file, 16000, y)
        
    print(file, max(y), min(y))

    return y

def split_meta(meta_list, ratio=0.05):

    num_test = int(len(meta_list) * ratio)
    
    random.shuffle(meta_list)
    
    meta_valid = meta_list[:num_test]
    meta_train = meta_list[num_test:]
    
    return meta_train, meta_valid 

def collate_function(meta):

    mels = list()
    emos = list()

    for m in meta:
        # 'ANAD/1sec_segmented_part3/1sec_segmented_part3/V7_1 (140).wav',
        # 'ANAD',
        # 'anger',
        # 0.09532879818594105

        y = load_wav(m[0])
        mel = MEL_LOADER.get_mel(y, 16000) # [80, 153]
        e = emotions.index(m[2])

        mels.append(mel.T) # [T, 80]
        emos.append(e)

    return pad_sequence(mels, batch_first=True).unsqueeze(1), torch.tensor(emos)

def get_meta_train_valid(json_file='hparams.json'):
    
    dataset_keywords = ['ACRYL', 'CREMA-D', 'EMOV', 'ShEMO', 
                    'BAVED', 'ANAD', 'TESS', 'EEKK',
                    'JL-corpus', 'RAVDESS', 'VIVAE', 'URDU',   
                    'CaFE', 'AESDD', 'SAVEE']
    
    meta = list(map(get_meta, dataset_keywords))
    
    meta_flat = list(itertools.chain.from_iterable(meta))

    meta_flat = list(filter(lambda x: x[2] in emotions, meta_flat))
    
    meta_train, meta_valid = split_meta(meta_flat)

    meta_train.sort(key=lambda x: x[-1])
    meta_valid.sort(key=lambda x: x[-1])

    return meta_train, meta_valid

def get_data_loader(batch_size=4, shuffle=False, num_workers=0,):
    
    meta_train, meta_valid = get_meta_train_valid()

    # meta_train.sort(key=lambda x: x[-1])
    meta_valid.sort(key=lambda x: x[-1])

    # https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206                            
    train_loader = DataLoader(meta_train,
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, # Change this value
                              collate_fn=collate_function)

    valid_loader = DataLoader(meta_valid,
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_function)

    return train_loader, valid_loader


if __name__ == "__main__":
        
    train_loader, valid_loader = get_data_loader()

    for d in train_loader:
        print(d[0].shape)
        print(d[1].shape)
        break

    for d in valid_loader:
        print(d[0].shape)
        print(d[1].shape)
        break

#     d = {'sampling_rate':16000,
#     'filter_length':1024,
#     'hop_length':256,
#     'win_length':1024,
#     'n_mel_channels':80,
#     'mel_fmin':0.0,
#     'mel_fmax':8000.0}
    
#     json.dump(d, open('hparams.json', 'w'), indent=4, sort_keys=True)