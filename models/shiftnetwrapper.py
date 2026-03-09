import torch
import torch.nn as nn
import sys
import os
import importlib.util
from pathlib import Path

class ShiftNet(nn.Module):
    """wrapper for shiftnet to work with our distillation setup"""
    
    def __init__(self):
        super(ShiftNet, self).__init__()
        # find repo root
        repo_root = Path(__file__).parent.parent
        
        # model paths
        shift_net_dir = repo_root / "Shift-Net"
        shift_net_arch_dir = shift_net_dir / "basicsr" / "models" / "archs"
        
        # try these model files in order
        model_files = [ 
            "gshift_denoise1.py",
            "gshift_denoise2.py"
        ]
        
        # Create model options (default options as used in make_model)
        opt = {'pretrain_models_dir': str(shift_net_dir / 'pretrained_models/')}
        
        # Try each model file until we find one that exists
        self.model = None
        for model_file in model_files:
            model_path = shift_net_arch_dir / model_file
            if model_path.exists():
                try:
                    print(f"Attempting to load ShiftNet model from {model_path}")
                    # Load the module containing make_model function
                    spec = importlib.util.spec_from_file_location("gshift_module", model_path)
                    gshift_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(gshift_module)
                    
                    # Initialize the ShiftNet model using make_model function
                    self.model = gshift_module.make_model(opt)
                    print(f"Successfully loaded ShiftNet model from {model_file}")
                    break
                except Exception as e:
                    print(f"Error loading model from {model_file}: {e}")
        
        if self.model is None:
            raise FileNotFoundError("Could not find a compatible ShiftNet model file")
    
    def load_checkpoint(self, checkpoint_path, device):
        """load weights from checkpoint, handles different formats"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if "params" in checkpoint:
            print("Found 'params' key in checkpoint, loading from params")
            state_dict = checkpoint["params"]
        elif "model" in checkpoint:
            print("Found 'model' key in checkpoint, loading from model")
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            print("Found 'state_dict' key in checkpoint, loading from state_dict")
            state_dict = checkpoint["state_dict"]
        else:
            print("No recognized keys found, attempting to load directly")
            state_dict = checkpoint
        
        # Try to load the state dict
        try:
            # Try direct loading first
            self.model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded checkpoint with non-strict matching")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Count matches and show some examples of keys
            model_keys = set(self.model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            print(f"Missing keys: {len(missing_keys)} keys")
            if missing_keys:
                print(f"Examples: {list(missing_keys)[:5]}")
            
            print(f"Unexpected keys: {len(unexpected_keys)} keys")
            if unexpected_keys:
                print(f"Examples: {list(unexpected_keys)[:5]}")
            
            # Try to load with strict=False again
            print("Attempting to load with strict=False")
            self.model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded checkpoint with non-strict matching")
    
    def forward(self, x, noise_map):
        """reshape input to match shiftnet's format and run forward pass"""
        B, total_channels, H, W = x.size()
        channels_per_frame = 3  # RGB images
        num_frames = total_channels // channels_per_frame
        center_idx = num_frames // 2
        
        # process one batch item at a time
        outputs = []
        
        for b in range(B):
            # reshape to [1, frames, C, H, W]
            single_x = x[b:b+1].view(1, num_frames, channels_per_frame, H, W)
            single_noise_map = noise_map[b:b+1]
            
            # expand noise map to match frames
            noise_map_expanded = single_noise_map.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)
            
            # run model
            with torch.no_grad():
                single_output = self.model(single_x, noise_map_expanded)
                
                # get center frame output
                if len(single_output.shape) == 5:  # [1, frames, C, H, W]
                    single_output = single_output[:, center_idx]  # [1, C, H, W]
                elif len(single_output.shape) == 4 and single_output.shape[0] == num_frames:
                    single_output = single_output[center_idx:center_idx+1]  # [1, C, H, W]
                
                # Add to outputs list
                outputs.append(single_output)
        
        # Concatenate all batch outputs
        output = torch.cat(outputs, dim=0)  # [B, C, H, W]
        
        return output