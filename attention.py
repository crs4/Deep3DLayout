import torch
############################################################################
# built upon HoHoNet implementation https://github.com/sunset1995/HoHoNet ##
############################################################################

import torch.nn as nn
import torch.nn.functional as F

import copy
import math


''' Transformer encoder '''
class TransformerEncoder(nn.Module):
    ''' Adapt from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py '''
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x

class TransformerEncoderLayer(nn.Module):
    ''' Adapt from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, mode='pre'):
        super(TransformerEncoderLayer, self).__init__()
        self.mode = mode
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.mode == 'post':
            x2 = self.self_attn(x, x, x)[0]
            x = x + self.dropout1(x2)
            x = self.norm1(x)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = x + self.dropout2(x2)
            x = self.norm2(x)
            return x
        elif self.mode == 'pre':
            x2 = self.norm1(x)
            x2 = self.self_attn(x2, x2, x2)[0]
            x = x + self.dropout1(x2)
            x2 = self.norm2(x)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
            x = x + self.dropout2(x2)
            return x
        raise NotImplementedError

class TransEn(nn.Module):
    def __init__(self, c_mid, position_encode, nhead=8, num_layers=2, dim_feedforward=2048, mode='pre'):
        super(TransEn, self).__init__()
        if isinstance(c_mid, (tuple, list)):
            c_mid = c_mid[0]
        encoder_layer = TransformerEncoderLayer(c_mid, nhead, dim_feedforward, mode=mode)
        self.transen = TransformerEncoder(encoder_layer, num_layers)

        self.out_channels = c_mid

        self.position_encode = position_encode ##default
        self.c_mid = c_mid

        ####TO DO use this and remove runtime instance in forward
        self.pos = self.positional_encoding(position_encode, c_mid)
        ##self.positional_encoding(position_encode, c_mid)  
      
    def positional_encoding(self, position_encode, c_mid):
            max_len, d_model = position_encode, c_mid
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            ##self.register_buffer('pos', pe.T[None].contiguous())
            #
            return pe.T[None].contiguous()    

    def forward(self, feat):        
        ##   
        feat = (feat + self.pos.to(feat.device)).permute(2,0,1)
        feat = self.transen(feat).permute(1,2,0)
        
        return feat

class MHSATransformerPos(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, maximum_position_encoding):
        super().__init__()
        
        self.encoder = TransEn(c_mid=d_model, position_encode=maximum_position_encoding, nhead=num_heads, num_layers=num_layers, dim_feedforward=conv_hidden_dim, mode='pre')
                        
    def forward(self, x):
        ##(batch_size, input_seq_len, d_model)
        x = x.permute(0, 2, 1) ##(batch_size, d_mode,input_seq_len)
        x = self.encoder(x)
        x = x.permute(0, 2, 1) ##(batch_size, input_seq_len, d_model)
               
        return x