import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools

from torch.autograd import Variable as V
from pytorch3d.structures import Meshes

from gcn import MeshRefinementHead

def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)

class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)

def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )        

#####resnet encoder from torchvision
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool
                
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4

####GAF encoding
class AConv(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3, st=(2, 1)):
        super(AConv, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=st, padding=ks//2),
            nn.ELU(inplace=True),
            )

    def forward(self, x):
        return self.layers(x)
     
class Slicing(nn.Module):
    def __init__(self, in_c, out_c, st=(2, 1), encoder_type = 'resnet18', interpolate = True):
        super(Slicing, self).__init__()
        ####3 filters-> height reduction by 8
        self.layer = nn.Sequential(
            AConv(in_c, in_c//2, st=st),
            AConv(in_c//2, in_c//4, st=st),
            AConv(in_c//4, out_c, st=st),
        )

        self.encoder_type = encoder_type

        self.interpolate = interpolate

    def forward(self, x, out_w):
        x = self.layer(x)
                        
        if( (x.shape[3] != out_w) and self.interpolate): 
            assert out_w % x.shape[3] == 0
            factor = out_w // x.shape[3]
            #####HorizonNet-style upsampling        
            x = torch.cat([x[..., -1:], x, x[..., :1]], 3) ## plus 2 on W
            x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False) ####NB interpolating only W
            x = x[..., factor:-factor] ##minus 2 on W           

        return x

class SplittedMultiSlicing(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8, backbone = 'resnet18', interpolate_feats = False, reshape_fh = True):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(SplittedMultiSlicing, self).__init__()
        self.cs = c1, c2, c3, c4 ##256 512 1024 2048 resnet50
        self.out_scale = out_scale
        
        self.interpolate_feats = interpolate_feats
                
        self.reshape_fh = reshape_fh                  
        
        self.slc_lst = nn.ModuleList([
            Slicing(c1, c1//out_scale, encoder_type = backbone, interpolate = interpolate_feats), ##256->32 resnet50 ##64->8 resnet18
            Slicing(c2, c2//out_scale, encoder_type = backbone, interpolate = interpolate_feats), ##512->64
            Slicing(c3, c3//out_scale, encoder_type = backbone, interpolate = interpolate_feats), ##1024->128
            Slicing(c4, c4//out_scale, encoder_type = backbone, interpolate = interpolate_feats), ##2048->256
        ])

    def forward(self, conv_list, out_w):
        ###out_w: must be the rnn sequence length
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        
        ###DEBUG
        feature = []
        for f, x in zip(self.slc_lst, conv_list):
            fs = x.shape[3]
                                               
            if(self.interpolate_feats):
                fs = out_w
            
            if(self.reshape_fh):
                feature.append(f(x, out_w).reshape(bs, -1, fs))
            else:
                feature.append(f(x, out_w))
                                
        return feature

####Deep3DLayout model
class Deep3DlayoutNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, decoder_type = 'conv', full_size = True):
        super(Deep3DlayoutNet, self).__init__()

        ###GAF support#########################################
        self.backbone = backbone        
        self.out_scale = 1     
        self._size = 512        
        self.full_size = full_size

        self.out_w_size = 512                            
        
        if(self.full_size):
            self.out_w_size = 1024
            
        self.c_last = self.out_w_size // 2 ### default h dim
        
        self.use_last = False 

        self.decoder_type = decoder_type

        self.mhsa_heads = 4                     
        #####################################################

        self.subdivide = True
                
        if(backbone == 'resnet18' or backbone == 'resnet50'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
            
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.c_last, self.out_w_size)##NB c1, c2, c3, c4 
                # Inference channels number from each block of the encoder
                c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)] ###NB depend by resnet layers depth 
                                                                               
                self.c_last = (c1*8 + c2*4 + c3*2 + c4*1) // 2 ####default: 1024
                                         
                                             
        # 1D prediction
        if(self.use_last):
            self.c_last = self.c_last // 4  ##                                      
                                       
        self.reshape_fh = False

        self.out_scale = 1

        self.slicing_module = SplittedMultiSlicing(c1, c2, c3, c4, self.out_scale, backbone = self.backbone, interpolate_feats = False, reshape_fh = self.reshape_fh)
            
        lfeats_dim = c1//self.out_scale+c2//self.out_scale+c3//self.out_scale+c4//self.out_scale                                              
                                                                                               
        self.p2m = MeshRefinementHead(input_channels = lfeats_dim, stage_depth = 6, use_mhsa = True, num_stages = 2, ico_sphere_level = 3, use_pos_encoding=True)

        self.subdivide = True
                     
        ''' Pad left/right-most to each other instead of zero padding '''       
        wrap_lr_pad(self)    
                   

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        x = self._prepare_x(x)

        conv_list = self.feature_extractor(x) #####ResNet by default           
           
        feature = self.slicing_module(conv_list, x.shape[3])            
                                      
        output = self.p2m(feature, meshes=None, subdivide=self.subdivide, reshaped_fh=self.reshape_fh)                       
                                                     
        return output

                     
def counter():
    print('testing Deep3DlayoutNet')

    from thop import profile, clever_format
        
    device = torch.device('cuda')

    net = Deep3DlayoutNet('resnet18').to(device)
            
    # testing
    rgb_inputs = [torch.randn(1, 3, 512, 1024).to(device)]
    
    with torch.no_grad():
        flops, params = profile(net, rgb_inputs)
    ##print(f'input :', [v.shape for v in inputs])
    print(f'flops : {flops/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        net(rgb_inputs[0])
        for _ in range(50):
            eps_time = time.time()
            net(rgb_inputs[0])
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')  


if __name__ == '__main__':
    counter()
        

    




