import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2  # For display and camera capture
from PIL import Image  # For converting numpy array to tensor
import time  # For FPS calculation
from collections import deque
from threading import Thread, Lock # Added Lock

# --- Import FastDVDnet model and utilities ---
from student import FastDVDnet, PocketDVDnet, PocketDVDnet7
from dataloaders.fastdvdnet.utils import remove_dataparallel_wrapper

def rgb_to_bgr(tensor):
    """
    Convert RGB image tensor to BGR.

    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (N, C, H, W) in RGB order.

    Returns:
        torch.Tensor: Image tensor in BGR order.
    """
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        return tensor[[2, 1, 0], :, :]
    elif tensor.dim() == 4 and tensor.shape[1] == 3:
        return tensor[:, [2, 1, 0], :, :]
    else:
        raise ValueError("Tensor must have shape (3, H, W) or (N, 3, H, W)")

# --- Threaded Camera Class ---
class ThreadedCamera:
    def __init__(self, src=0, cap_props=None):
        # Ensure src is correctly formatted for VideoCapture, especially if it's a path
        if isinstance(src, str) and src.startswith("/dev/video"):
            try:
                camera_index = int(src.replace("/dev/video", ""))
                self.stream = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            except ValueError:
                print(f"Warning: Could not parse camera index from {src}. Trying as is.")
                self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        else:
            self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)

        if cap_props:
            for prop, value in cap_props.items():
                self.stream.set(prop, value)

        if not self.stream.isOpened():
            raise RuntimeError(f"Error: Could not open camera {src}")

        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed and self.frame is None: # Initial read check
             raise RuntimeError(f"Error: Failed to grab initial frame from camera {src}")
        
        self.lock = Lock()
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(0.01) # Wait a bit before retrying
                continue
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            grabbed = self.grabbed
            frame = self.frame.copy() if self.grabbed and self.frame is not None else None
        return grabbed, frame

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()

    def get_properties(self):
        return {
            "width": int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.stream.get(cv2.CAP_PROP_FPS)
        }


def main(model):
    """
    Main processing loop for live camera stream denoising with FastDVDnet.
    Args:
        model (nn.Module): The FastDVDnet model loaded with pre-trained weights.
    """

    # Initialize video capture using ThreadedCamera
    cap_props = {
        cv2.CAP_PROP_FRAME_WIDTH: args.display_width,
        cv2.CAP_PROP_FRAME_HEIGHT: args.display_height,
        cv2.CAP_PROP_FPS: 30
    }
    try:
        threaded_cap = ThreadedCamera(src=f"/dev/video{args.camera_index}", cap_props=cap_props)
    except RuntimeError as e:
        print(e)
        exit()

    # Get actual camera properties
    props = threaded_cap.get_properties()
    print(f"Camera opened: {props['width']}x{props['height']} @ {props['fps']:.2f} FPS (requested {cap_props[cv2.CAP_PROP_FPS]} FPS)")

    print("Starting live camera stream processing with FastDVDnet...")
    print("Press \'q\' or ESC to quit.")

    # --- Frame Buffer ---
    frame_buffer = deque(maxlen=NUM_IN_FR_EXT)

    # --- FPS Calculation ---
    prev_time = time.time() # Initialize prev_time
    frame_count = 0
    fps_display = "FPS: 0"

    while True:
        ret, frame_bgr = threaded_cap.read()
        if not ret or frame_bgr is None:
            # print("Error: Failed to capture frame from threaded_cap.")
            time.sleep(0.01) # Wait a bit before retrying
            continue

        # --- Calculate FPS ---
        current_time = time.time()
        frame_count += 1
        elapsed_time = current_time - prev_time
        if elapsed_time >= 1.0:  # Update FPS display every second
            fps = frame_count / elapsed_time
            fps_display = f"FPS: {fps:.2f}"
            prev_time = current_time
            frame_count = 0
        
        # --- Preprocess Frame (on GPU) ---
        frame_gpu_hwc = torch.from_numpy(frame_bgr).to(args.device) # HWC, BGR, uint8 -> HWC, BGR, uint8 on GPU
        # Permute HWC to CHW, select BGR->RGB, convert to float, normalize, convert to half
        # Input frame_gpu_hwc is (H, W, C_bgr). After permute(2,0,1) it's (C_bgr, H, W).
        # BGR to RGB: R is index 2, G is index 1, B is index 0 for the channel dimension.
        # So, tensor_chw_bgr[[2,1,0],:,:] reorders channels to RGB.
        processed_frame_tensor = frame_gpu_hwc.permute(2, 0, 1)[[2, 1, 0], :, :].float().div_(255.0).half()
        frame_buffer.append(processed_frame_tensor)


        # --- Denoise only if buffer is full ---
        denoised_frame_display = None
        if len(frame_buffer) == NUM_IN_FR_EXT:
            # Stack frames in the buffer: (NUM_IN_FR_EXT, C_img, H_img, W_img)
            # frame_buffer already contains half-precision tensors on GPU
            stacked_frames_tensor = torch.stack(list(frame_buffer), dim=0) 
            _num_f, C_img, H_proc, W_proc = stacked_frames_tensor.shape

            # Reshape for model input: (1, NUM_IN_FR_EXT * C_img, H_proc, W_proc)
            # model_input_tensor is already half-precision
            model_input_tensor = stacked_frames_tensor.view(1, NUM_IN_FR_EXT * C_img, H_proc, W_proc)

            # Create noise map (constant for the whole sequence)
            noise_map_channels = 7 * 3  # 21 channels like in training
            noise_map = torch.tensor([args.noise_sigma], dtype=torch.float32, device=args.device) \
                        .expand(1, noise_map_channels, H_proc, W_proc).half()

            # model_input_tensor is already fp16
            # noise_map is now fp16
            
            # Perform denoising
            with torch.no_grad():
                out = torch.clamp(model(model_input_tensor, noise_map), 0, 1) 
            
            # Squeeze batch dimension (out is FP16)
            out_squeezed = out.squeeze(0) # CHW, RGB, float16, normalized, on GPU


            # Convert denoised tensor to displayable format (NumPy uint8, BGR)
            denoised_frame_display = (
                rgb_to_bgr(out_squeezed)          # (3, H, W) → BGR
                .mul(255)                                   # scale to [0, 255]
                .clamp(0, 255)                              # clamp values to valid range
                .to(dtype=torch.uint8)                      # convert to uint8 (efficient)
                .permute(1, 2, 0)                           # (3, H, W) → (H, W, 3)
                .cpu()
                .numpy()
            )

        # --- Prepare for Display ---
        # Left side: Processed center frame from buffer, converted for display
        if len(frame_buffer) == NUM_IN_FR_EXT:
            center_frame_tensor_gpu = frame_buffer[NUM_IN_FR_EXT // 2] # CHW, RGB, float16, normalized, on GPU
            center_frame_display = (
                rgb_to_bgr(center_frame_tensor_gpu)          # (3, H, W) → BGR
                .mul(255)                                   # scale to [0, 255]
                .clamp(0, 255)                              # clamp values to valid range
                .to(dtype=torch.uint8)                      # convert to uint8 (efficient)
                .permute(1, 2, 0)                           # (3, H, W) → (H, W, 3)
                .cpu()
                .numpy()
            )

        else:
            # Fallback if buffer not full: use current raw frame, convert to BGR if needed (it's already BGR)
            # To keep dimensions consistent for hstack if denoised_frame_display is None yet
            center_frame_display = frame_bgr.copy()

 
        # Create placeholder if denoised frame isn't ready yet
        if denoised_frame_display is None:
            denoised_frame_display = np.zeros_like(center_frame_display)
            cv2.putText(denoised_frame_display, "Buffering...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 1, cv2.LINE_AA)


        # Combine images for side-by-side display
        # Ensure both frames have the same height for hstack
        if center_frame_display.shape[0] != denoised_frame_display.shape[0] or \
           center_frame_display.shape[1] != denoised_frame_display.shape[1]:
            # This might happen if fallback center_frame_display (raw frame_bgr) has different dims
            # than the processed one (which denoised_frame_display is based on).
            # However, with current logic, they should match as no resizing is done post-capture.
            # If resizing were introduced, this would need careful handling.
            # For now, assume they match or make denoised_frame_display match center_frame_display if it's the placeholder.
            if np.all(denoised_frame_display == 0): # If it's the placeholder
                denoised_frame_display = np.zeros_like(center_frame_display)


        combined_display = np.hstack((center_frame_display, denoised_frame_display))

        # Display the combined image
        window_title = f"Live Camera (Center Frame) | FastDVDnet Denoised (Camera {args.camera_index}) - Press Q or ESC to quit"
        cv2.imshow(window_title, combined_display)

        # Update FPS display in-place only when it changes
        if hasattr(main, "_last_fps_display"):
            if fps_display != main._last_fps_display:
                print(f"\r{fps_display}", end="", flush=True)
                main._last_fps_display = fps_display
        else:
            print(f"\r{fps_display}", end="", flush=True)
            main._last_fps_display = fps_display

        # Wait for key press (1ms delay) and check if it's 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ESC key
            print("Quit signal received.")
            break

    # Release resources
    threaded_cap.stop() # Stop the threaded camera
    cv2.destroyAllWindows()
    print("Processing finished.")


if __name__ == "__main__":

    # --- PyTorch Optimizations ---
    torch.backends.cudnn.benchmark = True

    # --- Configuration ---
    # NUM_IN_FR_EXT = 7 # Number of frames required by pocketdvdnet

    class InferenceArgs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = 'pocket_models/7F_charb/34.14.pt'
        camera_index = 0
        noise_sigma = 30.0
        display_width = 1280
        display_height = 720
        num_input_frames = 7  # Number of frames in the sequence (5 or 7)


    args = InferenceArgs()
    args.noise_sigma /= 255. # Normalize noise sigma to [0, 1]

    print(f"Using device: {args.device}")
    print(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Using camera index: {args.camera_index}")
    print(f"Using noise sigma: {args.noise_sigma * 255.0}") # Print original sigma
    print(f"Using {args.num_input_frames} input frames")

    # --- Model Loading ---
    try:
        # Select the appropriate model based on num_input_frames
        if args.num_input_frames == 7:
            from student.pocketdvdnet7f import PocketDVDnet7
            model = PocketDVDnet7(num_input_frames=args.num_input_frames, num_color_ch=3, noise_ch_per_frame=3)
            print(f"Using PocketDVDnet7 model ({args.num_input_frames} frames)")
        elif args.num_input_frames == 5:
            from student.pocketdvdnet import PocketDVDnet
            model = PocketDVDnet(num_input_frames=args.num_input_frames, num_color_ch=3, noise_ch_per_frame=3)
            print(f"Using PocketDVDnet model ({args.num_input_frames} frames)")
        else:
            raise ValueError(f"Unsupported number of input frames: {args.num_input_frames}. Must be 5 or 7.")

        # Load saved weights
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        
        # Get the model's state dictionary from the checkpoint
        model_weights = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
        if not isinstance(model_weights, dict):
            pass 
        elif "model_state" in checkpoint:
            model_weights = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            model_weights = checkpoint["model_state_dict"]
        else:
            print("Warning: Could not find 'model_state_dict' or 'model_state' in checkpoint, assuming checkpoint IS the state_dict.")


        # If model was saved with DataParallel, remove the wrapper
        if "module." in list(model_weights.keys())[0]:
            model_weights = remove_dataparallel_wrapper(model_weights)

        # Load state dict
        model.load_state_dict(model_weights)

        # Move model to device and convert to half precision
        model.to(args.device)
        model.half()
        model.eval()
        model = torch.compile(model)
        print("PocketDVDnet model loaded successfully (FP16 enabled).")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    
    # Set NUM_IN_FR_EXT for main function based on args
    NUM_IN_FR_EXT = args.num_input_frames
    
    main(model)