import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.io import save_obj

from deep3dlayout_model import Deep3DlayoutNet

def_pth ='ckpt/m3d_layout.pth'

def_output_dir = 'results/'

def_img = 'input/UwV83HsGsw3_71ada030981d4468b76dcebc1b6fb940.png'  

def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net

def x2image(x):
    img = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default = def_pth, help='path to load saved checkpoint.')
    parser.add_argument('--img', required=False, default = def_img)
    parser.add_argument('--output_dir', required=False, default = def_output_dir)
    parser.add_argument('--visualize', action='store_true', default = True)
    parser.add_argument('--save_obj', action='store_true', default = True)
    
    args = parser.parse_args()
        
    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)

    device = 'cuda'

    # Loaded trained model
    net = load_trained_model(Deep3DlayoutNet, args.pth).to(device)
    net.eval()
    
    img_pil = Image.open(args.img)
    H, W = 512, 1024
    img_pil = img_pil.resize((W,H), Image.BICUBIC)
    img = np.array(img_pil, np.float32)[..., :3] / 255.

    x_img = torch.FloatTensor(img.transpose([2, 0, 1]).copy())

    with torch.no_grad():
        x = x_img.unsqueeze(0)
        x_img_c = x2image(x_img)                                                                                  
                  
        mesh_lev = -1 ####last level

        out_meshes = net(x.to(device)) 
        pred_mesh = out_meshes[mesh_lev].cpu().detach()             
                                                
        vertices = pred_mesh.verts_packed()
        triangles = pred_mesh.faces_packed()
                                         
       
        if(args.save_obj):
            ###                                                      
            head, tail = os.path.split(args.img)

            if not os.path.isdir(args.output_dir):
                print('Output directory %s not existed. Create one.' % args.output_dir)
                os.makedirs(args.output_dir)
                            
            f_name_pred = args.output_dir+tail[:-4]+'_pred.obj'
            save_obj(f_name_pred, vertices, triangles)
                    

        if(args.visualize):
            import open3d

            mesh3d = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles)
            )

            mesh3d.compute_vertex_normals()
        
            open3d.visualization.draw_geometries([mesh3d])   
                          
        plt.figure(0)
        plt.title('img')
        plt.imshow(x_img_c)            
                    
        plt.show()  


