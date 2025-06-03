"""
Definition of the FastDVDnet model
"""
import torch
import torch.nn as nn

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	# def __init__(self, num_in_frames, out_ch):
	def __init__(self, num_in_frames, out_ch, num_color_ch, num_noise_ch_for_concat): # MODIFIED
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			# nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, # ORIGINAL
			nn.Conv2d(num_in_frames*(num_color_ch + num_noise_ch_for_concat), num_in_frames*self.interm_ch, # MODIFIED
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)
		
	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames processed by this block (typically 3)
		num_color_ch: int. number of channels in input color frames (e.g., 3 for RGB)
		num_effective_noise_ch: int. number of channels in the noise_map argument to forward()
	Inputs of forward():
		xn: input frames of dim [N, num_color_ch, H, W]
		noise_map: array with noise map of dim [N, num_effective_noise_ch, H, W]
	"""

	# def __init__(self, num_input_frames=3): # ORIGINAL
	def __init__(self, num_input_frames=3, num_color_ch=3, num_effective_noise_ch=1): # MODIFIED
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		# self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0) # ORIGINAL
		self.inc = InputCvBlock(num_in_frames=num_input_frames, 
								out_ch=self.chs_lyr0, 
								num_color_ch=num_color_ch, 
								num_noise_ch_for_concat=num_effective_noise_ch) # MODIFIED
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		# self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3) # ORIGINAL
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=num_color_ch) # MODIFIED

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, num_color_ch, H, W] in the [0., 1.] range
			noise_map: Tensor [N, num_effective_noise_ch, H, W] in the [0., 1.] range
		'''
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

class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of constructor:
		num_input_frames: int. number of input frames for the model (e.g., 5)
		num_color_ch: int. number of channels in input color frames (e.g., 3 for RGB)
		noise_ch_per_frame: int. number of channels for the noise component corresponding to each frame
							 in the bundled noise_map input. If None, defaults to num_color_ch.
	Inputs of forward():
		x: input frames of dim [N, num_input_frames*num_color_ch, H, W]
		noise_map_bundle: array with noise map of dim [N, num_input_frames*noise_ch_per_frame, H, W]
	"""

	# def __init__(self, num_input_frames=5): # ORIGINAL
	def __init__(self, num_input_frames=5, num_color_ch=3, noise_ch_per_frame=None): # MODIFIED
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		self.num_color_ch = num_color_ch # ADDED

		if noise_ch_per_frame is None: # ADDED BLOCK
			self.noise_ch_per_frame_in_bundle = num_color_ch
		else:
			self.noise_ch_per_frame_in_bundle = noise_ch_per_frame

		# Define models of each denoising stage
		# DenBlock's num_input_frames is fixed at 3 (for its internal t-1, t, t+1 processing)
		# DenBlock will receive a noise map with self.noise_ch_per_frame_in_bundle channels.
		# self.temp1 = DenBlock(num_input_frames=3) # ORIGINAL
		# self.temp2 = DenBlock(num_input_frames=3) # ORIGINAL
		self.temp1 = DenBlock(num_input_frames=3, 
							  num_color_ch=self.num_color_ch, 
							  num_effective_noise_ch=self.noise_ch_per_frame_in_bundle) # MODIFIED
		self.temp2 = DenBlock(num_input_frames=3, 
							  num_color_ch=self.num_color_ch, 
							  num_effective_noise_ch=self.noise_ch_per_frame_in_bundle) # MODIFIED
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	# def forward(self, x, noise_map): # ORIGINAL
	def forward(self, x, noise_map_bundle): # MODIFIED
		'''Args:
			x: Tensor, [N, num_input_frames*num_color_ch, H, W] in the [0., 1.] range
			noise_map_bundle: Tensor [N, num_input_frames*noise_ch_per_frame_in_bundle, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		# (x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames)) # ORIGINAL
		frames = tuple(x[:, self.num_color_ch*m : self.num_color_ch*(m+1), :, :] 
					   for m in range(self.num_input_frames))

		# The current FastDVDnet structure with specific temp1 calls implies num_input_frames is 5.
		if self.num_input_frames != 5:
			raise ValueError(f"FastDVDnet forward pass is structured for num_input_frames=5, but got {self.num_input_frames}")

		x0, x1, x2, x3, x4 = frames[0], frames[1], frames[2], frames[3], frames[4]

		# Extract the noise map for the central frame (x2) from the bundle.
		# This single extracted noise map will be used for all DenBlock stages.
		# DenBlocks are initialized to expect a noise map with self.noise_ch_per_frame_in_bundle channels.
		center_frame_index_in_sequence = self.num_input_frames // 2 # e.g., index 2 for 5 frames
		
		start_channel_idx = center_frame_index_in_sequence * self.noise_ch_per_frame_in_bundle
		end_channel_idx = start_channel_idx + self.noise_ch_per_frame_in_bundle
		noise_map_for_denblocks = noise_map_bundle[:, start_channel_idx:end_channel_idx, :, :]

		# First stage
		# x20 = self.temp1(x0, x1, x2, noise_map) # ORIGINAL
		x20 = self.temp1(x0, x1, x2, noise_map_for_denblocks) # MODIFIED
		# x21 = self.temp1(x1, x2, x3, noise_map) # ORIGINAL
		x21 = self.temp1(x1, x2, x3, noise_map_for_denblocks) # MODIFIED
		# x22 = self.temp1(x2, x3, x4, noise_map) # ORIGINAL
		x22 = self.temp1(x2, x3, x4, noise_map_for_denblocks) # MODIFIED

		#Second stage
		# x = self.temp2(x20, x21, x22, noise_map) # ORIGINAL
		x = self.temp2(x20, x21, x22, noise_map_for_denblocks) # MODIFIED

		return x

