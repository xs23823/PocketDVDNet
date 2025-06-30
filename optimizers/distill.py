import os
import torch
import torch.nn as nn
import sys

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_output, teacher_output, ground_truth):
        gt_loss = self.mse_loss(student_output, ground_truth)
        #if torch.isnan(teacher_output).any():
            #return gt_loss
        distill_loss = self.mse_loss(student_output, teacher_output)
        return (1 - self.alpha) * gt_loss + self.alpha * distill_loss  

class PACNetTeacher:
    def __init__(self, device='cuda'):
        sys.path.append('./teachers/PaCNet')
        from modules import VidCnn, TfNet
        from functions import denoise_video_sequence
        
        self.device = device
        self.s_cnn = VidCnn()
        self.t_cnn = TfNet()
        self.denoise_fn = denoise_video_sequence
        self.sigma = 30 #default noise level (teacher model loads usig this)
        
        # Load model weights
        self.load_models(sigma= self.sigma)
        self.s_cnn.to(device)
        self.t_cnn.to(device)
        
        print("pacnet teacher loaded")
    
    def load_models(self, sigma):
        """Load pre-trained weights for both S-CNN and T-CNN models"""
        
        # load S-CNN weights
        state_file_s = f'./teachers/PaCNet/models/s_cnn_video/model_state_sig{self.sigma}.pt'
        if not os.path.isfile(state_file_s):
            raise FileNotFoundError(f"S-CNN model file not found: {state_file_s}")
        
        print(f"loading S-CNN model state '{state_file_s}'")
        model_state_s = torch.load(state_file_s, map_location=self.device)
        self.s_cnn.load_state_dict(model_state_s['state_dict'])
        self.s_cnn.to(self.device).eval()
        
        # load T-CNN weights  
        state_file_t = f'./teachers/PaCNet/models/t_cnn/model_state_sig{self.sigma}.pt'
        if not os.path.isfile(state_file_t):
            raise FileNotFoundError(f"T-CNN model file not found: {state_file_t}")
            
        print(f" loading T-CNN model state '{state_file_t}'")
        model_state_t = torch.load(state_file_t, map_location=self.device)
        self.t_cnn.load_state_dict(model_state_t['state_dict'])
        self.t_cnn.to(self.device).eval()
        
    
    def teacher_targets(self, imgn_train, noise_std_tensor):
        """
        Teacher denoises same images as student
        args:
            imgn_train: [B, 21, H, W]
            noise_std_tensor: [B, 1, 1, 1] - noise levels
        """
        B, _, H, W = imgn_train.shape
        # FIX: correct reshape and clamp to [0,1] for pacnet
        frames_7 = torch.clamp(imgn_train.view(B, 7, 3, H, W), 0, 1)
        teacher_targets = []
        
        for b in range(B):
            # FIX: transpose to [1, 3, 7, H, W] for pacnet
            single_vid = frames_7[b:b+1].transpose(1, 2)
            sigma = max(10, min(50, int(noise_std_tensor[b].item() * 255)))
            
            denoised_vid_t, _, _ = self.denoise_fn(
                single_vid, f'batch_{b}', sigma,
                s_cnn=self.s_cnn, t_cnn=self.t_cnn,
                clipped_noise=False, gpu_usage=1, silent=True
            )
            
            center_frame = denoised_vid_t[0, :, 3, :, :]
            teacher_targets.append(center_frame)
        
        return torch.stack(teacher_targets, dim=0).to(self.device)


