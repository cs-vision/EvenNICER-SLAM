import numpy as np
import torch 
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
# import wandb

# TODO: unoffset and unscale (0.1?, but probly not because the decoder is supervised with real scale) the predicted translation as post processing
# maybe do it outside the model

class PoseGrid_decoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pe = config['PoseGrid']['positional_encoding']
        if self.pe:
            self.input_t_dim = config['PoseGrid']['PoseNet_freq'] * 2 + 96
        else:
            self.input_t_dim = 96
        self.define_network(config)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = config['tracking']['device']

    def define_network(self, config):
        layers_list = config['PoseGrid']['layers_feat']  
        L = list(zip(layers_list[:-1], layers_list[1:]))  

        # init TransNet
        self.mlp_transNet = torch.nn.ModuleList()
        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = self.input_t_dim
            if li in self.config['PoseGrid']['skip']: k_in += self.input_t_dim
            if li == len(L) - 1: k_out = 3
            linear = torch.nn.Linear(k_in, k_out)
            self.mlp_transNet.append(linear)
        
        # init QuatsNet
        self.mlp_quatsNet = torch.nn.ModuleList()
        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = self.input_t_dim
            if li in self.config['PoseGrid']['skip']: k_in += self.input_t_dim
            if li == len(L) - 1: k_out = 4
            linear = torch.nn.Linear(k_in, k_out)
            self.mlp_quatsNet.append(linear)     

    def forward(self, input_time, t1, t2, pose_encoding1, pose_encoding2, velocity1, velocity2):
        # Compute positional encoding with respect to t1 and t2 for input_time
        alpha = self.normalize_time(input_time, t1, t2)
        if self.pe:
            normalized_time = 2 * alpha - 1
            encoding_time = self.positional_encoding(normalized_time.unsqueeze(-1), L=self.config['PoseGrid']['PoseNet_freq'])
            input_trans1 = torch.cat([pose_encoding1[:, :96], encoding_time], dim=-1)
            input_quat1 = torch.cat([pose_encoding1[:, 96:], encoding_time], dim=-1)
            input_trans2 = torch.cat([pose_encoding2[:, :96], encoding_time], dim=-1)
            input_quat2 = torch.cat([pose_encoding2[:, 96:], encoding_time], dim=-1)
        else:
            input_trans1 = pose_encoding1[:, :96]
            input_quat1 = pose_encoding1[:, 96:]
            input_trans2 = pose_encoding2[:, :96]
            input_quat2 = pose_encoding2[:, 96:]

        if torch.equal(t1, t2):
            # If t1 equals t2, just decode once
            pred_translation = self.forward_transNet(input_trans1)
            pred_quaternion = self.forward_quatsNet(input_quat1)
            return pred_translation, pred_quaternion

        # Decode pose_encodings into translations and rotations
        pred_translation1 = self.forward_transNet(input_trans1)
        pred_quaternion1 = self.forward_quatsNet(input_quat1)
        pred_translation2 = self.forward_transNet(input_trans2)
        pred_quaternion2 = self.forward_quatsNet(input_quat2)

        # Cubic Hermite interpolation for translation
        interpolated_translation = []
        for i in range(3):
            interpolated_translation.append(self.cubic_hermite(input_time, t1, t2, 
                                                               pred_translation1[:, i], 
                                                               pred_translation2[:, i], 
                                                               velocity1[:, i], 
                                                               velocity2[:, i]))
        interpolated_translation = torch.stack(interpolated_translation, dim=-1)
        
        # Ensure quaternions are normalized
        pred_quaternion1 = F.normalize(pred_quaternion1, p=2, dim=-1)
        pred_quaternion2 = F.normalize(pred_quaternion2, p=2, dim=-1)

        # Squad interpolation for quaternion rotation
        a, b = self.compute_squad_intermediates(pred_quaternion1, pred_quaternion2, 
                                                velocity1[:, 3:], velocity2[:, 3:])
        interpolated_quaternion = self.quat_squad(pred_quaternion1, a, b, pred_quaternion2, alpha)

        return interpolated_translation, interpolated_quaternion

    def forward_transNet(self, trans_input):
        translation_feat = trans_input
        for li, layer in enumerate(self.mlp_transNet):
            if li in self.config['PoseGrid']['skip']: translation_feat = torch.cat([translation_feat, trans_input],dim=-1)
            translation_feat = layer(translation_feat)
            if li==len(self.mlp_transNet)-1:
                # last layer
                pass
            else:
                translation_feat = F.leaky_relu(translation_feat) 
        return translation_feat

    def forward_quatsNet(self, rot_input):
        rotation_feat = rot_input
        for li, layer in enumerate(self.mlp_quatsNet):
            if li in self.config['PoseGrid']['skip']: rotation_feat = torch.cat([rotation_feat, rot_input],dim=-1)
            rotation_feat = layer(rotation_feat)
            if li==len(self.mlp_quatsNet)-1:
                # last layer
                rotation_feat = F.tanh(rotation_feat)
            else:
                rotation_feat = F.leaky_relu(rotation_feat) 
        return rotation_feat

    def positional_encoding(self, input, L): 
        shape = input.shape
        freq = 2**torch.arange(L, dtype=torch.float32, device = self.device) * np.pi 
        spectrum = input[...,None] * freq 
        sin,cos = spectrum.sin(), spectrum.cos() 
        input_enc = torch.stack([sin,cos], dim=-2) 
        input_enc = input_enc.view(*shape[:-1], -1) 
        return input_enc

    def normalize_time(self, t, t1, t2):
        # normalize t to [0, 1]
        if torch.equal(t1, t2):
            return torch.zeros_like(t)
        l = (t - t1) / (t2 - t1)
        assert torch.all(l >= 0) and torch.all(l <= 1), "t is not between t1 and t2!"
        return l

    def hermite_basis(self, t, t1, t2):
        l = self.normalize_time(t, t1, t2)
        h00 = (1 + 2*l)*(1 - l)**2
        h10 = l*(1 - l)**2
        h01 = l**2*(3 - 2*l)
        h11 = l**2*(l - 1)
        return h00, h10, h01, h11

    def cubic_hermite(self, t, t1, t2, f1, f2, df1, df2):
        h00, h10, h01, h11 = self.hermite_basis(t, t1, t2)
        return h00*f1 + h10*df1*(t2 - t1) + h01*f2 + h11*df2*(t2 - t1)

    def _validate_unit(self, q):
        # A helper function to validate if a quaternion is unit
        norm = torch.norm(q, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm)), f"Quaternion must be unit. The norm is {norm}"

    def quat_exp(self, q):
        norms = torch.norm(q[..., 1:], dim=-1)
        e = torch.exp(q[..., 0])
        exp_q = torch.empty_like(q)
        exp_q[..., 0] = e * torch.cos(norms)

        # Handling the case where norms are zero
        norm_zero = torch.isclose(norms, torch.zeros_like(norms))
        not_zero = ~norm_zero

        exp_q[not_zero, 1:] = e[not_zero].unsqueeze(-1) * (q[not_zero, 1:] / norms[not_zero].unsqueeze(-1)) * torch.sin(norms[not_zero].unsqueeze(-1))
        exp_q[norm_zero, 1:] = 0.0

        return exp_q

    def quat_log(self, q):
        log_q = torch.empty_like(q)

        # We need all the norms to avoid divide by zeros later.
        # Can also use these to minimize the amount of work done.
        q_norms = torch.norm(q, dim=-1)
        q_norm_zero = torch.isclose(q_norms, torch.zeros_like(q_norms))
        q_not_zero = ~q_norm_zero
        v_norms = torch.norm(q[..., 1:], dim=-1)
        v_norm_zero = torch.isclose(v_norms, torch.zeros_like(v_norms))
        v_not_zero = ~v_norm_zero

        if torch.any(q_not_zero):
            log_q[q_not_zero, 0] = torch.log(q_norms[q_not_zero])
            if torch.any(q_norm_zero):
                log_q[q_norm_zero, 0] = -float('inf')
        else:
            log_q[..., 0] = -float('inf')

        if torch.any(v_not_zero):
            prefactor = q[v_not_zero, 1:] / v_norms[v_not_zero].unsqueeze(-1)

            inv_cos = torch.acos(q[v_not_zero, 0] / q_norms[v_not_zero])

            if torch.any(v_norm_zero):
                log_q[v_norm_zero, 1:] = 0.0
            log_q[v_not_zero, 1:] = prefactor * inv_cos.unsqueeze(-1)
        else:
            log_q[..., 1:] = 0.0

        return log_q

    def quat_conjugate(self, q):
        conjugate_q = torch.clone(q)
        conjugate_q[..., 1:] *= -1
        return conjugate_q

    def quat_multiply(self, qi, qj):
        output = torch.empty_like(qi)
        output[..., 0] = qi[..., 0] * qj[..., 0] - torch.sum(qi[..., 1:] * qj[..., 1:], dim=-1)
        qi = qi.to(torch.float32)
        qj = qj.to(torch.float32)
        output[..., 1:] = qi[..., 0].unsqueeze(-1) * qj[..., 1:] + qj[..., 0].unsqueeze(-1) * qi[..., 1:] + torch.cross(qi[..., 1:], qj[..., 1:])
        return output

    def quat_power(self, q, n):
        # Use broadcasting to make q and n shape-compatible
        newshape = torch.broadcast_shapes(q[..., 0].shape, n.shape)
        q = q.expand(*newshape, 4)
        n = n.expand(*newshape)
        
        # Note that we follow the convention that 0^0 = 1
        check = n == 0
        if torch.any(check):
            powers = torch.empty(*newshape, 4)
            powers[check] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            not_check = ~check
            if torch.any(not_check):
                powers = powers.to(self.device)
                powers[not_check] = self.quat_exp(n[not_check].unsqueeze(-1) * self.quat_log(q[not_check, :]))
        else:
            powers = powers.to(self.device)
            powers = self.quat_exp(n.unsqueeze(-1) * self.quat_log(q))

        return powers

    def quat_slerp(self, q0, q1, t, ensure_shortest=True):
        self._validate_unit(q0)
        self._validate_unit(q1)
        t = torch.clamp(t, 0, 1)

        q0 = torch.as_tensor(torch.atleast_2d(q0))
        q1 = torch.tensor(torch.atleast_2d(q1))

        # Ensure that we turn the short way around
        if ensure_shortest:
            cos_theta = torch.sum(q0 * q1, dim=-1)
            flip = cos_theta < 0
            q1[flip] *= -1

        return self.quat_multiply(q0, self.quat_power(self.quat_multiply(self.quat_conjugate(q0), q1), t))

    def quat_squad(self, p, a, b, q, t):
        self._validate_unit(p)
        self._validate_unit(a)
        self._validate_unit(b)
        self._validate_unit(q)
        t = torch.clamp(t, 0, 1)

        return self.quat_slerp(
            self.quat_slerp(p, q, t, ensure_shortest=False),
            self.quat_slerp(a, b, t, ensure_shortest=False),
            2 * t * (1 - t),
            ensure_shortest=False,
        )

    def quat_normalize(self, q):
        norms = torch.norm(q, dim=-1)
        return q / norms.unsqueeze(-1)

    def compute_squad_intermediates(self, q0, q1, q0_prime, q1_prime):
        # Compute the conjugate (inverse) of q0 and q1
        q0_inv = self.quat_conjugate(q0)
        q1_inv = self.quat_conjugate(q1)
        
        # Some multiplications
        q0inv_q1 = self.quat_multiply(q0_inv, q1)
        q0inv_q0prime = self.quat_multiply(q0_inv, q0_prime)
        q1inv_q1prime = self.quat_multiply(q1_inv, q1_prime)

        # Some substractions
        delta_q0 = q0inv_q0prime - self.quat_log(q0inv_q1)
        delta_q1 = q1inv_q1prime - self.quat_log(q0inv_q1)

        # Compute the exponentials for a and b
        exp_q0 = self.quat_exp(0.5 * delta_q0)
        exp_q1 = self.quat_exp(-0.5 * delta_q1)

        # Compute a and b
        a = self.quat_multiply(q0, exp_q0)
        b = self.quat_multiply(q1, exp_q1)

        # normalize a and b just in case
        a = self.quat_normalize(a)
        b = self.quat_normalize(b)

        return a, b