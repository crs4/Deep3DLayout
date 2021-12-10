##################################################################################
#Graph Convolutional Network implementation built upon Mesh R-CNN and PyTorch3D ##
##################################################################################

#BSD License

#For meshrcnn software

#Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#Redistribution and use in source and binary forms, with or without modification,
#are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name Facebook nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn
from torch.nn import functional as F

from pytorch3d.ops import GraphConv, SubdivideMeshes, sample_points_from_meshes, vert_align
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from gaf import gravity_projection

def _padded_to_packed(x, idx):
    D = x.shape[-1]
    idx = idx.view(-1, 1).expand(-1, D)
        
    x_packed = x.view(-1, D).gather(0, idx.to(x.device))

    return x_packed

def ico_vertex_count(level):
    v_count = 12
    for i in range(level):
        v_count = (v_count*4)-6

    return v_count

class MeshRefinementHead(nn.Module):
    def __init__(self, input_channels, num_stages = 2, hidden_dim = 128, stage_depth = 3, graph_conv_init = 'normal', use_activation = False, 
                 use_mhsa = False, use_rnn = False, mhsa_num_heads = 4, mhsa_hdim_factor = 2, use_bn_mhsa = False, ico_sphere_level = 2, use_pos_encoding = False, reduce_feats = True):
        super(MeshRefinementHead, self).__init__()

        self.num_stages = num_stages

        self.ico_sphere_level = ico_sphere_level
        
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            vert_feat_dim = 0 if i == 0 else hidden_dim
            stage = MeshRefinementStage(
                input_channels, vert_feat_dim, hidden_dim, stage_depth, gconv_init=graph_conv_init, 
                use_mhsa = use_mhsa, use_rnn = use_rnn, mhsa_num_heads = mhsa_num_heads, mhsa_hdim_factor = mhsa_hdim_factor, use_pos_encoding = use_pos_encoding, base_sphere_level = ico_sphere_level, level = i, reduce_feats = reduce_feats)
            self.stages.append(stage)

        self.use_activation = use_activation
           
    def forward(self, img_feats, meshes, subdivide=False, reshaped_fh=False):       
                
        meshes = ico_sphere(self.ico_sphere_level, img_feats[0].device)
        meshes = meshes.extend(img_feats[0].shape[0])
               
        output_meshes = []

        vert_feats = None

        for i, stage in enumerate(self.stages):
            meshes, vert_feats = stage(img_feats, meshes, vert_feats, use_activation = self.use_activation, reshaped_fh=reshaped_fh)
                                    
            output_meshes.append(meshes)

            if subdivide and i < self.num_stages - 1:
                subdivide = SubdivideMeshes()
                meshes, vert_feats = subdivide(meshes, feats=vert_feats)
                              
        return output_meshes


class MeshRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, stage_depth, reduce_feats, gconv_init="normal", use_mhsa = False, use_rnn = False, mhsa_num_heads = 4, mhsa_hdim_factor = 2, 
                 use_pos_encoding=False, base_sphere_level = 3, level = 0):
        
        super(MeshRefinementStage, self).__init__()               
        
        self.vert_offset = nn.Linear(hidden_dim + 3, 3)

        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3

            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)
                   
        
        self.mhsa_linear_reduction = False

        self.reduce_feats = reduce_feats

        if(self.reduce_feats):
            self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)
            # initialization for bottleneck and vert_offset
            nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.vert_offset.weight)
        nn.init.constant_(self.vert_offset.bias, 0)

        v_count = ico_vertex_count(base_sphere_level+level)
                                                       
        self.projection = gravity_projection(use_mhsa = use_mhsa, use_rnn = use_rnn, lfeats = img_feat_dim, num_heads = mhsa_num_heads, hdim_factor = mhsa_hdim_factor, 
                                             use_pos_encoding=use_pos_encoding, verts_count = v_count)
        

    def forward(self, img_feats, meshes, vert_feats=None, use_activation = False, reshaped_fh=False):
        
        verts_padded_to_packed_idx = meshes.verts_padded_to_packed_idx()                               
        vert_pos_padded = meshes.verts_padded()
                       
        vert_pos_packed = _padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
                        
        vert_align_feats = self.projection(img_feats, vert_pos_padded, get_vertices = False, is_squeezed_h = reshaped_fh)
                                                 
                       
        vert_align_feats = _padded_to_packed(vert_align_feats, verts_padded_to_packed_idx) 
        
        if(self.reduce_feats):
            vert_align_feats = F.relu(self.bottleneck(vert_align_feats))
                       
        first_layer_feats = [vert_align_feats, vert_pos_packed]                
        
        if vert_feats is not None:
            first_layer_feats.append(vert_feats)
                                          
        vert_feats = torch.cat(first_layer_feats, dim=1)
              
        ep = meshes.edges_packed()

        # Run graph conv layers
        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, ep))
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)
        
        # Predict a new mesh by offsetting verts
         
        if(use_activation): ####if normalized mesh
            vert_offsets = torch.tanh(self.vert_offset(vert_feats))
        else:
            vert_offsets = self.vert_offset(vert_feats)        
        
        meshes = meshes.offset_verts(vert_offsets)
        
        return meshes, vert_feats_nopos 

