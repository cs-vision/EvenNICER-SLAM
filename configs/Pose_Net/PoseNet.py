
# This PoseNet code shows the implemention of ours continous pose representation
# In the experiment there might be some small difference

import torch
import torch.nn.functional as torch_F

# example configs
config = {
        'PoseNet_freq': 5,
        'layers_feat': [None,256,256,256,256,256,256,256,256],
        'min_time': 0,
        'max_time': 100
        }
        
class PoseNet(torch.nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.input_t_dim = config['PoseNet_freq'] * 2 + 1
        self.define_network(config)

        self.min_time = config['min_time']
        self.max_time = config['max_time']


    def define_network(self,config):
        layers_list = config['layers_feat']  
        L = list(zip(layers_list[:-1],layers_list[1:]))  

        # init TransNet
        self.mlp_transNet = torch.nn.ModuleList()
        
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in cfg['tracking']['skip'] : k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.mlp_transNet.append(linear)
        
        # init QuatsNet
        self.mlp_quatsNet = torch.nn.ModuleList()
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in [4]: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 4
            linear = torch.nn.Linear(k_in,k_out)
            self.mlp_quatsNet.append(linear)
            
    def forward(self, input_time):
        # normalization to (-1, 1)
        input_time = 2*(input_time - self.min_time)/(self.max_time - self.min_time) - 1
        encoding_time = self.positional_encoding(input_time, L = self.config['PoseNet_freq'] )
        encoding_time = torch.cat([input_time,encoding_time],dim=-1) 
        
        pred_translation = self.forward_transNet(encoding_time)
        pred_quaternion =  self.forward_quatsNet(encoding_time)

        return pred_translation, pred_quaternion

    def forward_transNet(self, encoding_time)
    
        translation_feat = encoding_time
        for li, layer in enumerate(self.mlp_transNet):
            if li in self.cfg['tracking']['skip']: translation_feat = 
            torch.cat([translation_feat,encoding_time],dim=-1)
            translation_feat = layer(translation_feat)
            if li==len(self.mlp_transnet)-1:
                # last layer
                translation_feat = translation_feat
            else:
                translation_feat = torch_F.leaky_relu(translation_feat) 
        return translation_feat

    def forward_quatsNet(self, encoding_time)
    
        rotation_feat = encoding_time
        for li, layer in enumerate(self.mlp_quatsNet):
            if li in self.cfg['tracking']['skip']: rotation_feat = 
            torch.cat([rotation_feat,encoding_time],dim=-1)
            rotation_feat = layer(rotation_feat)
            if li==len(self.mlp_quatsNet)-1:
                # last layer
                rotation_feat = torch_F.tanh(rotation_feat)
            else:
                rotation_feat = torch_F.leaky_relu(rotation_feat) 
        return rotation_feat

    def cal_jacobian(self, input_time):
        input_time = torch.tensor(input_time.clone(), requires_grad=True)
        input_time = 2*(input_time - self.min_time)/(self.max_time - self.min_time) - 1
        encoding_time = self.positional_encoding(input_time, 
        L = self.config['PoseNet_freq'] )
        encoding_time = torch.cat([input_time,encoding_time],dim=-1) 

        translation_jacobian = torch.autograd.functional.jacobian(
        self.forward_transNet, encoding_time)
        rotation_jacobian = torch.autograd.functional.jacobian(
        self.forward_quatsNet, encoding_time)

        return translation_jacobian, rotation_jacobian

    def positional_encoding(self,input, L): 
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device = self.device)*np.pi 
        spectrum = input[...,None]*freq 
        sin,cos = spectrum.sin(),spectrum.cos() 
        input_enc = torch.stack([sin,cos],dim=-2) 
        input_enc = input_enc.view(*shape[:-1],-1) 
        return input_enc
