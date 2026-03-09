import argparse
import time
import cv2
import torch
from tqdm import tqdm
import torch.nn as nn
import os
from models.fastdvdnet import FastDVDnet
from models.pocketdvdnet7f import PocketDVDnet7 
from models.pocketdvdnet import PocketDVDnet
from dataloaders.fastdvdnet.utils import *
from torchvision.utils import make_grid

# Will be set by CLI argument instead of constant
# NUM_IN_FR_EXT = 7
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir, ('n{}_{}').format(sigmaval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir, ('n{}_{}').format(sigmaval, idx) + fext)
		else:
			out_name = os.path.join(save_dir, ('n{}_{}_{}').format(sigmaval, suffix, idx) + fext)
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
			img_combined = cv2.vconcat([noisyimg, outimg])
			cv2.imwrite(out_name, img_combined)
		else:
			outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
			cv2.imwrite(out_name, outimg)

def test_fastdvdnet(**args):
	start_time = time.time()
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	device = torch.device('cuda' if not args['no_gpu'] and torch.cuda.is_available() else 'cpu')
	print('Loading model ...')
	
	# Select model based on sequence length
	num_input_frames = args['num_input_frames']
	if num_input_frames == 7:
		model_temp = PocketDVDnet7(num_input_frames=num_input_frames).to(device)
		print(f"Using PocketDVDnet7 with {num_input_frames} input frames")
	elif num_input_frames == 5:
		model_temp = PocketDVDnet(num_input_frames=num_input_frames).to(device)
		print(f"Using PocketDVDnet with {num_input_frames} input frames")
	else:
		raise ValueError(f"Unsupported number of input frames: {num_input_frames}. Must be 5 or 7.")
	
	state_temp_dict = torch.load(args['model_file'], map_location=device, weights_only=False)
	if "module." in list(state_temp_dict.keys())[0]:
		state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
	model_temp.load_state_dict(state_temp_dict['model_state'])
	model_temp.eval()
	with torch.no_grad():
		seq, _, _ = open_sequence(args['test_path'],
									args['gray'],
									expand_if_needed=False,
									max_num_fr=args['max_num_fr_per_seq'])
		seq = torch.from_numpy(seq).to(device)
		seq_time = time.time()
		noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
		seqn = seq + noise
		noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)
		denframes = denoise_seq_fastdvdnet(seq=seqn,
										noise_std=noisestd,
										temp_psz=num_input_frames,
										model_temporal=model_temp)
	stop_time = time.time()
	psnr = batch_psnr(denframes, seq, 1.)
	psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)
	loadtime = (seq_time - start_time)
	runtime = (stop_time - seq_time)
	seq_length = seq.size()[0]
	print("Finished denoising {}".format(args['test_path']))
	print("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s".format(seq_length, runtime, loadtime))
	print("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))
	if not args['dont_save_results']:
		save_out_seq(seqn, denframes, args['save_path'],
					   int(args['noise_sigma']*255), args['suffix'], args['save_noisy'])
	
	return psnr  # Return PSNR for averaging

def is_folder_only(path):
	# Returns True if all items in path are folders (and at least one folder exists)
	return os.path.isdir(path) and any(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path)) \
		and all(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path) if not f.startswith('.'))

def find_all_leaf_dirs(root):
	# Recursively find all leaf directories (directories that do not contain any subdirectories)
	leaf_dirs = []
	for dirpath, dirnames, filenames in os.walk(root):
		# Ignore hidden folders
		dirnames[:] = [d for d in dirnames if not d.startswith('.')]
		if not dirnames:  # No subdirectories
			leaf_dirs.append(dirpath)
	return leaf_dirs

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Denoise a sequence with PocketDVDnet")
	parser.add_argument("--model_file", type=str,
						default="/home/imogend/Documents/Data/pocketdvd checkpoints/34.3.pt",
						help='path to model of the pretrained denoiser')
	parser.add_argument("-i","--test_path", type=str, default="/home/imogend/Documents/Data/DAVIS 420p JPEG VAL",
						help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=100,
						help='max number of frames to load per sequence')
	parser.add_argument("--num_input_frames", type=int, default=5,
						help='number of input frames (5 or 7)')
	parser.add_argument("--noise_sigma", type=float, default=50, help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("-o","--save_path", type=str, default='./results',
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',
						help='perform denoising of grayscale images instead of RGB')

	argspar = parser.parse_args()
	argspar.noise_sigma /= 255.

	print(f"\n### Testing {'PocketDVDnet7Frame' if argspar.num_input_frames == 7 else 'PocketDVDnet'} model ###")

	# Check if test_path contains only folders (recursively process all leaf dirs)
	if is_folder_only(argspar.test_path):
		leaf_dirs = find_all_leaf_dirs(argspar.test_path)
		for leaf in tqdm(leaf_dirs, desc="Processing folders"):
			# Compute relative path to preserve structure
			rel_path = os.path.relpath(leaf, argspar.test_path)
			save_dir = os.path.join(argspar.save_path, rel_path)
			os.makedirs(save_dir, exist_ok=True)
			args_dict = vars(argspar).copy()
			args_dict['test_path'] = leaf
			args_dict['save_path'] = save_dir
			psnr = test_fastdvdnet(**args_dict)
			if 'total_psnr' not in locals():
				total_psnr = []
			total_psnr.append(psnr)
		
		# Calculate and display average PSNR for all sequences
		avg_psnr = sum(total_psnr) / len(total_psnr)
		print(f"\nAverage PSNR across all {len(total_psnr)} sequences: {avg_psnr:.4f}dB")
	else:
		test_fastdvdnet(**vars(argspar))