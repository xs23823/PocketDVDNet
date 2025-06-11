import torch
import torch.nn as nn

class PocketCvBlock(nn.Module):
    '''Compressed (Conv2d => BN => ReLU) x 2 with exact dimensions'''
    def __init__(self, in_ch, mid_ch, out_ch):
        super(PocketCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),      # convblock.0
            nn.BatchNorm2d(mid_ch),                                              # convblock.1
            nn.ReLU(inplace=True),                                               # convblock.2
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),    # convblock.3
            nn.BatchNorm2d(out_ch),                                              # convblock.4
            nn.ReLU(inplace=True)                                                # convblock.5
        )

    def forward(self, x):
        return self.convblock(x)

class PocketInputCvBlock(nn.Module):
    '''Compressed Input Block: 18 → 90 → 16'''
    def __init__(self, num_in_frames, num_color_ch, num_noise_ch_for_concat):
        super(PocketInputCvBlock, self).__init__()
        # Input: num_in_frames * (num_color_ch + num_noise_ch_for_concat) = 3 * (3 + 3) = 18
        self.convblock = nn.Sequential(
            # convblock.0: (18, 90) from layer_specs
            nn.Conv2d(18, 90, kernel_size=3, padding=1, groups=num_in_frames, bias=False),
            nn.BatchNorm2d(90),                                                  # convblock.1
            nn.ReLU(inplace=True),                                               # convblock.2
            # convblock.3: (90, 16) from layer_specs  
            nn.Conv2d(90, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),                                                  # convblock.4
            nn.ReLU(inplace=True)                                                # convblock.5
        )

    def forward(self, x):
        return self.convblock(x)

class PocketDownBlock(nn.Module):
    '''Compressed Downscale + CvBlock'''
    def __init__(self, in_ch, stride_out_ch, cvblock_mid_ch, cvblock_out_ch):
        super(PocketDownBlock, self).__init__()
        self.convblock = nn.Sequential(
            # convblock.0: stride conv
            nn.Conv2d(in_ch, stride_out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(stride_out_ch),                                       # convblock.1
            nn.ReLU(inplace=True),                                               # convblock.2
            # convblock.3: CvBlock
            PocketCvBlock(stride_out_ch, cvblock_mid_ch, cvblock_out_ch)
        )

    def forward(self, x):
        return self.convblock(x)

class PocketUpBlock(nn.Module):
    '''Compressed CvBlock + Upscale'''
    def __init__(self, in_ch, cvblock_mid_ch, cvblock_out_ch, final_out_ch):
        super(PocketUpBlock, self).__init__()
        self.convblock = nn.Sequential(
            # convblock.0: CvBlock  
            PocketCvBlock(in_ch, cvblock_mid_ch, cvblock_out_ch),
            # convblock.1: conv before PixelShuffle
            nn.Conv2d(cvblock_out_ch, final_out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)                                                   # convblock.2
        )
        
    def forward(self, x):
        return self.convblock(x)

class PocketOutputCvBlock(nn.Module):
    '''Compressed Output Block: 16 → 32 → 3'''
    def __init__(self, in_ch, out_ch):
        super(PocketOutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            # convblock.0: (16, 32) from layer_specs
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),                                                  # convblock.1
            nn.ReLU(inplace=True),                                               # convblock.2
            # convblock.3: (32, 3) from layer_specs
            nn.Conv2d(32, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)

class PocketDenBlock(nn.Module):
    """Compressed denoising block with exact layer_specs dimensions"""
    
    def __init__(self, num_input_frames=3, num_color_ch=3, num_effective_noise_ch=1):
        super(PocketDenBlock, self).__init__()
        
        # Build layers with exact compressed dimensions from layer_specs:
        
        # Input: 18 → 90 → 16
        self.inc = PocketInputCvBlock(num_in_frames=num_input_frames, 
                                     num_color_ch=num_color_ch, 
                                     num_noise_ch_for_concat=num_effective_noise_ch)
        
        # DownBlock0: 16 → 32, CvBlock: 32 → 32 → 32  
        self.downc0 = PocketDownBlock(in_ch=16, stride_out_ch=32, 
                                     cvblock_mid_ch=32, cvblock_out_ch=32)
        
        # DownBlock1: 32 → 64, CvBlock: 64 → 64 → 64
        self.downc1 = PocketDownBlock(in_ch=32, stride_out_ch=64,
                                     cvblock_mid_ch=64, cvblock_out_ch=64)
        
        # UpBlock2: CvBlock: 64 → 64 → 64, then 64 → 128 → PixelShuffle → 32
        self.upc2 = PocketUpBlock(in_ch=64, cvblock_mid_ch=64, cvblock_out_ch=64, final_out_ch=32)
        
        # UpBlock1: CvBlock: 32 → 32 → 32, then 32 → 64 → PixelShuffle → 16  
        self.upc1 = PocketUpBlock(in_ch=32, cvblock_mid_ch=32, cvblock_out_ch=32, final_out_ch=16)
        
        # Output: 16 → 32 → 3
        self.outc = PocketOutputCvBlock(in_ch=16, out_ch=num_color_ch)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        # Input convolution block
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)

        # Residual
        x = in1 - x

        return x

class PocketDVDnet(nn.Module):
    """Standalone Compressed FastDVDnet model with exact layer_specs dimensions hardcoded"""
    
    def __init__(self, num_input_frames=5, num_color_ch=3, noise_ch_per_frame=None):
        super(PocketDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        self.num_color_ch = num_color_ch

        if noise_ch_per_frame is None:
            self.noise_ch_per_frame_in_bundle = num_color_ch
        else:
            self.noise_ch_per_frame_in_bundle = noise_ch_per_frame

        # Create compressed denoising blocks with exact layer_specs dimensions
        self.temp1 = PocketDenBlock(num_input_frames=3, 
                                   num_color_ch=self.num_color_ch, 
                                   num_effective_noise_ch=self.noise_ch_per_frame_in_bundle)
        self.temp2 = PocketDenBlock(num_input_frames=3, 
                                   num_color_ch=self.num_color_ch, 
                                   num_effective_noise_ch=self.noise_ch_per_frame_in_bundle)
        
        # Initialize weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map_bundle):
        # Unpack inputs
        frames = tuple(x[:, self.num_color_ch*m : self.num_color_ch*(m+1), :, :] 
                       for m in range(self.num_input_frames))

        if self.num_input_frames != 5:
            raise ValueError(f"PocketDVDnet forward pass is structured for num_input_frames=5, but got {self.num_input_frames}")

        x0, x1, x2, x3, x4 = frames[0], frames[1], frames[2], frames[3], frames[4]

        # Extract noise map for central frame
        center_frame_index_in_sequence = self.num_input_frames // 2
        start_channel_idx = center_frame_index_in_sequence * self.noise_ch_per_frame_in_bundle
        end_channel_idx = start_channel_idx + self.noise_ch_per_frame_in_bundle
        noise_map_for_denblocks = noise_map_bundle[:, start_channel_idx:end_channel_idx, :, :]

        # First stage
        x20 = self.temp1(x0, x1, x2, noise_map_for_denblocks)
        x21 = self.temp1(x1, x2, x3, noise_map_for_denblocks)
        x22 = self.temp1(x2, x3, x4, noise_map_for_denblocks)

        # Second stage
        x = self.temp2(x20, x21, x22, noise_map_for_denblocks)

        return x

