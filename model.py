import torch
import torch.nn as nn
import torch.nn.functional as F

class ReferenceEncoder(nn.Module):

    def __init__(self, conv_h, gru_h, num_emo):
        super(ReferenceEncoder, self).__init__()

        module_container = list()
        for h_in, h_out in zip([1] + conv_h[:-1], conv_h):
            module_container.extend([nn.Conv2d(h_in, h_out, kernel_size=3, stride=2, padding=1), 
                                     nn.ReLU(),
                                     nn.BatchNorm2d(h_out)])

        self.conv_layers = nn.Sequential(*module_container)
        
        self.gru = nn.GRU(conv_h[-1] * 2, gru_h, batch_first=True)
        # input of shape (batch, seq_len, input_size)

        self.fc = nn.Linear(gru_h, num_emo)

    def forward(self, input_tensor):
        # tensor = self.conv(input_tensor)
        # tensor = F.relu(tensor)
        # tensor = self.batch_norm(tensor)
        tensor = self.conv_layers(input_tensor) # (B, H, T, M) 
        tensor = tensor.transpose(1, 2) # (B, T, H, M[2]) 

        B, T, H, M = tensor.shape
        tensor = tensor.reshape(B, T, -1) # (B, T, H * M) 
 
        tensor, h_n = self.gru(tensor) # (L, N, H), (1, N, H)
        h_n.squeeze_(0) # (1, N, H) => (N, H)

        h_n = torch.tanh(h_n)

        pred = F.log_softmax(self.fc(h_n), dim=-1)

        return pred

    def infer(self, input_tensor):
        # tensor = self.conv(input_tensor)
        # tensor = F.relu(tensor)
        # tensor = self.batch_norm(tensor)
        tensor = self.conv_layers(input_tensor) # (B, H, T, M) 
        tensor = tensor.transpose(1, 2) # (B, T, H, M[2]) 

        B, T, H, M = tensor.shape
        tensor = tensor.reshape(B, T, -1) # (B, T, H * M) 
 
        tensor, h_n = self.gru(tensor) # (L, N, H), (1, N, H)
        h_n.squeeze_(0) # (1, N, H) => (N, H)

        h_n = torch.tanh(h_n)

        pred = F.log_softmax(self.fc(h_n), dim=-1)

        return h_n, pred


class ReferenceEncoderMod(nn.Module):

    def __init__(self, conv_h, gru_h, num_emo):
        super(ReferenceEncoderMod, self).__init__()

        module_container = list()
        for h_in, h_out in zip([1] + conv_h[:-1], conv_h):
            module_container.extend([nn.Conv2d(h_in, h_out, kernel_size=3, stride=2, padding=1), 
                                     nn.ReLU(),
                                     nn.BatchNorm2d(h_out)])

        self.conv_layers = nn.Sequential(*module_container)
        
        self.gru_layers = nn.GRU(conv_h[-1] * 2, conv_h[-1] * 2, num_layers=3, batch_first=True)

        self.gru = nn.GRU(conv_h[-1] * 2, gru_h, batch_first=True)
        # input of shape (batch, seq_len, input_size)

        self.fc = nn.Linear(gru_h, num_emo)

    def forward(self, input_tensor):
        tensor = self.conv_layers(input_tensor) # (B, H, T, M) 
        tensor = tensor.transpose(1, 2) # (B, T, H, M[2]) 

        B, T, H, M = tensor.shape
        tensor = tensor.reshape(B, T, -1) # (B, T, H * M) 
 
        tensor, h_n = self.gru_layers(tensor)

        tensor, h_n = self.gru(tensor) # (L, N, H), (1, N, H)
        h_n.squeeze_(0) # (1, N, H) => (N, H)

        h_n = torch.tanh(h_n)

        pred = F.log_softmax(self.fc(h_n), dim=-1)

        return pred

    
    def infer(self, input_tensor):
        tensor = self.conv_layers(input_tensor) # (B, H, T, M) 
        tensor = tensor.transpose(1, 2) # (B, T, H, M[2]) 

        B, T, H, M = tensor.shape
        tensor = tensor.reshape(B, T, -1) # (B, T, H * M) 
 
        tensor, h_n = self.gru_layers(tensor)

        tensor, h_n = self.gru(tensor) # (L, N, H), (1, N, H)
        h_n.squeeze_(0) # (1, N, H) => (N, H)

        h_n = torch.tanh(h_n)

        pred = F.log_softmax(self.fc(h_n), dim=-1)

        return h_n, pred