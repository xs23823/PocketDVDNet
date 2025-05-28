import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2  # For display and camera capture
from PIL import Image  # For converting numpy array to tensor
import time  # For FPS calculation
from collections import deque

# --- Import FastDVDnet model and utilities ---
from models.fastdvdnet import FastDVDnet
from dataloaders.fastdvdnet.utils import remove_dataparallel_wrapper

# --- PyTorch Optimizations ---
# torch.backends.cudnn.benchmark = True

# --- Configuration ---
NUM_IN_FR_EXT = 5 # Number of frames required by FastDVDnet

class InferenceArgs:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint_path = './out/out_extra_data_3/best.pt'
	camera_index = 4
	noise_sigma = 30.0
	display_width = 1920
	display_height = 1080


args = InferenceArgs()
args.noise_sigma /= 255. # Normalize noise sigma to [0, 1]

print(f"Using device: {args.device}")
print(f"Loading checkpoint: {args.checkpoint_path}")
print(f"Using camera index: {args.camera_index}")
print(f"Using noise sigma: {args.noise_sigma * 255.0}") # Print original sigma

# --- Model Loading ---
try:
	model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

	# Load saved weights
	checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
	
	# Get the model's state dictionary from the checkpoint
	# Common keys are "model_state_dict" or "model_state". Adjust if your checkpoint uses a different key.
	model_weights = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
	if not isinstance(model_weights, dict):
		# If checkpoint is directly the state_dict
		pass # model_weights is already the state_dict
	elif "model_state" in checkpoint: # Handling the specific key mentioned in original code
		model_weights = checkpoint["model_state"]
	elif "model_state_dict" in checkpoint:
		model_weights = checkpoint["model_state_dict"]
	else:
		# Fallback if the structure is unknown but it's a dict; assumes it might be the state_dict itself
		# This might need adjustment based on the actual checkpoint structure if primary keys aren't found.
		print("Warning: Could not find 'model_state_dict' or 'model_state' in checkpoint, assuming checkpoint IS the state_dict.")


	# If model was saved with DataParallel, remove the wrapper
	if "module." in list(model_weights.keys())[0]:
		model_weights = remove_dataparallel_wrapper(model_weights)

	# Load state dict
	model.load_state_dict(model_weights)

	# Move model to device and convert to half precision
	model.to(args.device)
	model.half()  # Convert model to FP16

	# Set model to evaluation mode
	model.eval()
	model = torch.compile(model)
	print("FastDVDnet model loaded successfully (FP16 enabled).")

except FileNotFoundError:
	print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
	exit()
except Exception as e:
	print(f"Error loading model: {e}")
	exit()

# --- Image Processing Utilities ---
# Basic transform to tensor
transform = transforms.ToTensor()

# --- Main Processing Loop ---

# Initialize video capture
cap = cv2.VideoCapture(f"/dev/video{args.camera_index}", cv2.CAP_V4L2)
# Try setting resolution and FPS (may depend on camera capabilities)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.display_height)
cap.set(cv2.CAP_PROP_FPS, 60) # Lower FPS might be more stable

if not cap.isOpened():
	print(f"Error: Could not open camera {args.camera_index}")
	exit()

# Get actual camera properties
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")


print("Starting live camera stream processing with FastDVDnet...")
print("Press 'q' or ESC to quit.")

# --- Frame Buffer ---
frame_buffer = deque(maxlen=NUM_IN_FR_EXT)

# --- FPS Calculation ---
prev_time = 0
frame_count = 0
fps_display = "FPS: calculating..."

while True:
	ret, frame_bgr = cap.read()
	if not ret:
		print("Error: Failed to capture frame.")
		time.sleep(0.1) # Wait a bit before retrying
		continue

	# --- Calculate FPS ---
	frame_count += 1
	current_time = time.time()
	elapsed_time = current_time - prev_time
	if elapsed_time > 1.0:  # Update FPS display every second
		fps = frame_count / elapsed_time
		fps_display = f"FPS: {fps:.2f}"
		prev_time = current_time
		frame_count = 0

	# --- Preprocess Frame ---
	# # Resize frame for consistent display and potentially faster processing
	# h_orig, w_orig, _ = frame_bgr.shape
	# scale = args.display_width / w_orig
	# target_h = int(h_orig * scale)
	# # Ensure target dimensions are even (sometimes required by models)
	# target_w = args.display_width
	# if target_h % 2 != 0:
	# 	target_h -= 1
	# if target_w % 2 != 0:
	# 	target_w -= 1 # Should already be even if display_width is
	# frame_resized_bgr = cv2.resize(, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

	# Convert BGR frame to RGB PIL Image -> Tensor
	img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	img_tensor = transform(img_rgb).to(args.device) # (C, H, W), range [0, 1]

	# Add the current frame tensor to the buffer
	frame_buffer.append(img_tensor)

	# --- Denoise if buffer is full ---
	denoised_frame_display = None
	if len(frame_buffer) == NUM_IN_FR_EXT:
		# Stack frames in the buffer: (NUM_IN_FR_EXT, C_img, H_img, W_img)
		stacked_frames_tensor = torch.stack(list(frame_buffer), dim=0) 
		_num_f, C_img, H_proc, W_proc = stacked_frames_tensor.shape 

		# Reshape for model input: (1, NUM_IN_FR_EXT * C_img, H_proc, W_proc)
		model_input_tensor = stacked_frames_tensor.view(1, NUM_IN_FR_EXT * C_img, H_proc, W_proc)

		# Create noise map (constant for the whole sequence)
		# Model expects noise_map_bundle: [N, num_input_frames * noise_ch_per_frame_in_bundle, H, W]
		# noise_ch_per_frame_in_bundle defaults to num_color_ch (C_img)
		noise_map_channels = NUM_IN_FR_EXT * C_img
		noise_map = torch.tensor([args.noise_sigma], dtype=torch.float32, device=args.device) \
						 .expand(1, noise_map_channels, H_proc, W_proc)

		# Convert inputs to FP16 for the model
		model_input_tensor_fp16 = model_input_tensor.half()
		noise_map_fp16 = noise_map.half()

		# Perform denoising
		with torch.no_grad():
			# out will be FP16
			print("model_input_tensor_fp16 shape:", model_input_tensor_fp16.shape)
			out = torch.clamp(model(model_input_tensor_fp16, noise_map_fp16), 0, 1) 
		
		# Squeeze batch dimension (out is FP16)
		out_squeezed = out.squeeze(0)


		# Convert denoised tensor to displayable format (NumPy uint8, BGR)
		# Convert back to FP32 before CPU transfer for wider compatibility with NumPy/OpenCV operations
		denoised_frame_display = out_squeezed.float().permute(1, 2, 0).cpu().numpy() 
		denoised_frame_display = (denoised_frame_display * 255).astype(np.uint8)
		denoised_frame_display = cv2.cvtColor(denoised_frame_display, cv2.COLOR_RGB2BGR) # Convert RGB to BGR
		# print(denoised_frame_display)

	# --- Prepare for Display ---
	# Original center frame (resized BGR) - use the one from the buffer if available
	center_frame_display = frame_bgr # Default to current resized frame
	if len(frame_buffer) == NUM_IN_FR_EXT:
		center_frame_index = NUM_IN_FR_EXT // 2
		center_frame_tensor_display = frame_buffer[center_frame_index].permute(1, 2, 0).cpu().numpy()
		center_frame_display = (center_frame_tensor_display * 255).astype(np.uint8)
		center_frame_display = cv2.cvtColor(center_frame_display, cv2.COLOR_RGB2BGR)


	# Add FPS to the original display
	font_scale = 0.75
	font_thickness = 1
	font_color = (0, 255, 0) # Green
	cv2.putText(center_frame_display, fps_display, (10, args.display_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
				font_scale, font_color, font_thickness, cv2.LINE_AA)

	# Create placeholder if denoised frame isn't ready yet
	if denoised_frame_display is None:
		denoised_frame_display = np.zeros_like(center_frame_display)
		cv2.putText(denoised_frame_display, "Buffering...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


	# Combine images for side-by-side display
	combined_display = np.hstack((center_frame_display, denoised_frame_display))

	# Display the combined image
	window_title = f"Live Camera (Center Frame) | FastDVDnet Denoised (Camera {args.camera_index}) - Press Q or ESC to quit"
	cv2.imshow(window_title, combined_display)

	# Wait for key press (1ms delay) and check if it's 'q' or ESC
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q') or key == 27:  # 27 is the ESC key
		print("Quit signal received.")
		break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Processing finished.")