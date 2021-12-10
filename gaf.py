import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from attention import MHSATransformerPos

def xy2uv(xyz, eps = 0.001):
    x, y, z = torch.unbind(xyz, dim=2)

    x = x+eps
    y = y+eps
    z = z+eps
            
    u = torch.atan2(x, -y)
    v = - torch.atan(z / torch.sqrt(x**2 + y**2)) ###  (default: - for z neg (under horizon) - grid sample instead expects -1,-1 top-left
        
    pi = float(np.pi)

    u = u / pi
    v = (2.0 * v) / pi 

    u = torch.clamp(u, min=-1, max=1)
    v = torch.clamp(v, min=-1, max=1)
        
    ###output: [batch_size x num_points x 2]##range -1,+1

    output = torch.stack([u, v], dim=-1) 
                        
    return output

class gravity_projection(nn.Module):        
    def __init__(self, lfeats = 1024, use_mhsa = False, use_rnn = False, num_heads = 4, hdim_factor = 2, use_pos_encoding = False, verts_count = 642):
        super(gravity_projection, self).__init__()
        
        self.use_mhsa = use_mhsa
        self.lfeats = lfeats

        self.use_rnn = use_rnn
                
        if(self.use_mhsa):
            self.num_heads=num_heads
            self.mhsa = MHSATransformerPos(num_layers=1, d_model=self.lfeats, num_heads=num_heads, conv_hidden_dim=2048, maximum_position_encoding = verts_count)
        
        if(self.use_rnn):
            self.bi_rnn = nn.LSTM(input_size=self.lfeats,
                                  hidden_size=(self.lfeats//2),
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=False,
                                  bidirectional=True)

        self.drop_out = nn.Dropout(0.5)
    
    def slice_projection(self, uv_inputs, img_feature):      
        
        uv_inputs = uv_inputs.to(img_feature.device)                    
        uv_inputs = uv_inputs.unsqueeze(1)            
                
        output = F.grid_sample(img_feature, uv_inputs, align_corners=True)        
        output = torch.transpose(output.squeeze(2), 1, 2)                  
              
        return output
        
    def forward(self, img_features, inputs, is_squeezed_h = False, get_vertices = True, return_packed=False):
        ###
        uv_inputs = xy2uv(inputs) ####mesh device
                                               
        feats = []

        for img_feature in img_features:
            feats.append( self.slice_projection(uv_inputs, img_feature) )
                    
        output = torch.cat(feats, 2) 
                
        if(self.use_mhsa):
            output = self.mhsa(output)
            output = self.drop_out(output)
            
        if(self.use_rnn):
            output = output.permute(1, 0, 2)             
            output,hidden = self.bi_rnn(output)
            output = self.drop_out(output)
            output = output.permute(1, 0, 2)               
                 
        ###NB prepend previous state vertices coords
        if(get_vertices):
            output = torch.cat((inputs,output), 2) #### BxVx(1024+3)

        if(return_packed):
            output = output.view(-1, output.shape[-1])
                            
        return output


    



    

