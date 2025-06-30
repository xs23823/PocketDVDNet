#!/bin/sh
"""
Denoise	all	the	sequences existent in a	given folder using DVDnet.

@author: Matias	Tassano	<mtassano@parisdescartes.fr>
"""
import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn	as nn
from teachers import DVDnet_spatial, DVDnet_temporal, denoise_seq_dvdnet, load_dvdnet_models
from teachers.DVDnet.utils import *

NUM_IN_FRAMES =	5 #	temporal size of patch
MC_ALGO	= 'DeepFlow' # motion estimation algorithm
OUTIMGEXT =	'.png' # output	images format

def	save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
	"""Saves the denoised and noisy	sequences under	save_dir
	"""
	seq_len	= seqnoisy.size()[0]
	for	idx	in range(seq_len):
		# Build	Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('n{}_{}').format(sigmaval,	idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('n{}_DVDnet_{}').format(sigmaval, idx)	+ fext)
		else:
			out_name = os.path.join(save_dir,\
					('n{}_DVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name,	noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)

def	test_dvdnet(**args):
	"""Denoises	all	sequences present in a given folder. Sequences must	be stored as numbered
	image sequences. The different sequences must be stored	in subfolders under	the	"test_path"	folder.

	Inputs:
		args (dict)	fields:
			"model_spatial_file": path to model	of the pretrained spatial denoiser
			"model_temp_file": path	to model of the	pretrained temporal	denoiser
			"test_path": path to sequence to denoise
			"suffix": suffix to add	to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used	on test	set
			"dont_save_results:	if True, don't save	output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
	"""
	start_time = time.time()

	# If save_path does	not	exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args['save_path'])

	# Sets data	type according to CPU or GPU modes
	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	#load
	model_temp, model_spa = load_dvdnet_models(args['model_temp_file'], args['model_spatial_file'], num_input_frames=NUM_IN_FRAMES, device=device)

	with torch.no_grad():
		# process data
		seq, _, _ =	open_sequence(args['test_path'],\
									False,\
									expand_if_needed=False,\
									max_num_fr=args['max_num_fr_per_seq'])
		seq	= torch.from_numpy(seq[:, np.newaxis, :, :, :]).to(device)

		seqload_time = time.time()

		# Add noise
		noise =	torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
		seqn = seq + noise
		noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

		denframes =	denoise_seq_dvdnet(seq=seqn,
										noise_std=noisestd,
										temp_psz=NUM_IN_FRAMES,
										model_temporal=model_temp,
										model_spatial=model_spa,
										mc_algo=MC_ALGO)
		den_time = time.time()

	# Compute PSNR and log it
	psnr = batch_psnr(denframes, seq.squeeze(),	1.)
	psnr_noisy = batch_psnr(seqn.squeeze(),	seq.squeeze(), 1.)
	print("\tPSNR on {} : {}\n".format(os.path.split(args['test_path'])[-1], psnr))
	print("\tDenoising time: {:.2f}s".format(den_time -	seqload_time))
	print("\tSequence loaded in : {:.2f}s".format(seqload_time - start_time))
	print("\tTotal time: {:.2f}s\n".format(den_time	- start_time))
	logger.info("%s, %s, PSNR noisy	%fdB, PSNR %f dB" %	\
			 (args['test_path'], args['suffix'], psnr_noisy, psnr))
	# Save outputs
	if not args['dont_save_results']:
		# Save sequence 
		save_out_seq(seqn, denframes, args['save_path'], int(args['noise_sigma']*255), \
					   args['suffix'], args['save_noisy'])

	# close	logger
	close_logger(logger)

if __name__	== "__main__":
	parser = argparse.ArgumentParser(description="Denoise a	sequence with DVDnet")
	parser.add_argument("--model_temp_file", type=str,
						default="teachers/DVDnet/model_temp.pt",
						help='path to temporal model')
	parser.add_argument("--model_spatial_file",	type=str,
						default="teachers/DVDnet/model_spatial.pt",	
						help='path to spatial model')
	parser.add_argument("-i","--test_path",	type=str, 
						default="/home/imogend/Documents/Data/DAVIS 420p JPEG VAL/deer",
						help='path to sequence to denoise')
	parser.add_argument("--suffix",	type=str, default="", help='suffix to add to output	name')
	parser.add_argument("--max_num_fr_per_seq",	type=int, default=100,
						help='max number of frames to load per sequence')
	parser.add_argument("--noise_sigma", type=float, default=90, help='noise level used	on test	set')
	parser.add_argument("--dont_save_results", action='store_true',	help="don't	save output	images")
	parser.add_argument("--save_noisy",	action='store_true', help="save	noisy frames")
	parser.add_argument("--no_gpu",	action='store_true', help="run model on CPU")
	parser.add_argument("-o","--save_path",	type=str, default='./results',
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',
						help='perform denoising	of grayscale images	instead	of RGB')

	argspar = parser.parse_args()
	argspar.noise_sigma /= 255.
	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing DVDnet	model ###")
	print("> Parameters:")
	for	p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_dvdnet(**vars(argspar))
