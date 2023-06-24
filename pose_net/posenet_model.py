# TODO : deifine transNet, rotsNet, quaternion, pose

import torch
import torch.nn.functional as torch_F
import numpy as np

class PoseNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.posenet_freq = cfg['Pose_Net']['freq']
        self.input_t_dim = self.posenet_freq*2 + 1
        self.define_network(cfg)

        self.min_time = cfg['Pose_Net']['min_time']
        self.max_time = cfg['Pose_Net']['max_time']
    
    def positional_encoding(self, input, L):
        device = f'cuda:{input.get_device()}'
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32, device=device)*np.pi 
        spectrum = input[...,None]*freq 
        sin,cos = spectrum.sin(), spectrum.cos() 
        input_enc = torch.stack([sin,cos],dim=-2) 
        input_enc = input_enc.view(*shape[:-1],-1) 
        return input_enc
    

class transNet(PoseNet):
    def __init__(self, cfg):
        super().__init__(cfg)

    def define_network(self, cfg):
        self.layers_list = cfg['Pose_Net']['layers_feat']
        L = list(zip(self.layers_list[:-1],self.layers_list[1:]))  

        # init TransNet
        self.mlp_transNet = torch.nn.ModuleList()
        
        for li,(k_in,k_out) in enumerate(L):
            if li==0: 
                k_in = self.input_t_dim
            #if li in cfg['tracking']['skip'] : k_in += self.input_t_dim
            if li==len(L)-1:
                k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.mlp_transNet.append(linear)
    
    def forward(self, input_time):
        device = f'cuda:{input_time.get_device()}'
        # normalization to (-1, 1)
        input_time = 2*(input_time - self.min_time)/(self.max_time - self.min_time) - 1
        encoding_time = self.positional_encoding(input_time, L = self.posenet_freq)
        encoding_time = torch.cat([input_time, encoding_time], dim = -1)

        translation_feat = encoding_time
        mlp_transNet = self.mlp_transNet.to(device)
        for li, layer in enumerate(mlp_transNet):
            translation_feat = layer(translation_feat)
            if li != len(self.mlp_transNet) - 1:
                translation_feat = torch_F.leaky_relu(translation_feat)
            # last layer
            else:
                pred_translation = translation_feat
        return pred_translation

class quatsNet(PoseNet):
    def __init__(self, cfg):
        super().__init__(cfg)

    def define_network(self, cfg): 
        self.layers_list = cfg['Pose_Net']['layers_feat']
        L = list(zip(self.layers_list[:-1],self.layers_list[1:]))  

        # init TransNet
        self.mlp_quatsNet = torch.nn.ModuleList()
        
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            #if li in cfg['tracking']['skip'] : k_in += self.input_t_dim
            if li==len(L)-1: k_out = 4
            linear = torch.nn.Linear(k_in,k_out)
            self.mlp_quatsNet.append(linear)
        # self.mlp_quatsNet.to(self.device)

    def forward(self, input_time):
        device = f'cuda:{input_time.get_device()}'
        # normalization to (-1, 1)
        input_time = 2*(input_time - self.min_time)/(self.max_time - self.min_time) - 1
        encoding_time = self.positional_encoding(input_time, L = self.posenet_freq)
        encoding_time = torch.cat([input_time, encoding_time], dim = -1)

        quaternion_feat = encoding_time
        mlp_quatsNet = self.mlp_quatsNet.to(device)
        for li, layer in enumerate(mlp_quatsNet):
            quaternion_feat = layer(quaternion_feat)
            if li != len(self.mlp_quatsNet) - 1:
                quaternion_feat = torch_F.leaky_relu(quaternion_feat)
            # last layer
            else:
                pred_quaternion = quaternion_feat
        return pred_quaternion


# class quatsNet(PoseNet):
#     def __init__(self, cfg):
#         super().__init__(cfg)
        
#         for li, (k_in, k_out) in enumerate(self.L):
#             if li == 0:
#                 k_in = self.input_t_dim
#             if li == len(self.L) - 1:
#                 k_out = 4
#             linear = torch.nn.Linear(k_in, k_out)
#             self.mlp_quatsNet.append(linear)
    
#     def forward(self, input_time):
#         # normalization to (-1, 1)
#         input_time = 2*(input_time - self.min_time)/(self.max_time - self.min_time) - 1
#         encoding_time = self.positional_encoding(input_time, L = self.posenet_freq)
#         encoding_time = torch.cat([input_time, encoding_time], dim = -1)

#         quaternion_feat = encoding_time
#         for li, layer in enumerate(self.mlp_quatsNet):
#             quaternion_feat = layer(quaternion_feat)
#             if li != len(self.mlp_quatsNet) - 1:
#                 quaternion_feat = torch_F.leaky_relu(quaternion_feat)
#             # last layer
#             else:
#                 pred_quaternion = quaternion_feat
#         return pred_quaternion
