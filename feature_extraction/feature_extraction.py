import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import load_file
import hydra
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import shutil

def check_disk_space(required_bytes, path='.'):
    """Check if disk has enough space"""
    stat = shutil.disk_usage(path)
    free_space = stat.free
    return free_space > required_bytes

def visualize_single_frame_features(feature_map, save_path, frame_idx, title="Feature Map", 
                                   cmap='viridis', colorbar=True, dpi=150):
    """
    Visualize single frame feature map (based on channel average)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate channel average
    mean_feat = np.mean(feature_map, axis=0)
    
    # Normalize to 0-1 range
    min_val = mean_feat.min()
    max_val = mean_feat.max()
    if max_val > min_val:
        norm_feat = (mean_feat - min_val) / (max_val - min_val)
    else:
        norm_feat = mean_feat
    
    # Create visualization image
    plt.figure(figsize=(10, 8))
    plt.imshow(norm_feat, cmap=cmap)
    if colorbar:
        plt.colorbar()
    plt.title(f"{title} (Frame {frame_idx})", fontsize=12)
    plt.axis('off')
    
    # Save image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close()
    return save_path

def parse_args():
    parser = argparse.ArgumentParser(description='Extract video features using GFE model')

    # Required arguments
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the input video file')

    # Model configuration
    parser.add_argument('--config_dir', type=str, default='./configs',
                        help='Directory containing model config files')
    parser.add_argument('--model_config_path', type=str,
                        default='./configs/sd21_feature_extractor.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--ckpt_path', type=str,
                        default='./pre_train_model/gfe_sd21_full.safetensors',
                        help='Path to model checkpoint file')

    # Processing parameters
    parser.add_argument('--num_frames', type=int, default=41182,
                        help='Number of frames to process')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing frames')
    parser.add_argument('--feat_key', type=str, default='mid',
                        help='Feature layer to use (e.g., mid, up, down)')
    parser.add_argument('--caption', type=str, default='A video frame',
                        help='Caption/description for the video frame')

    # Image processing parameters
    parser.add_argument('--image_size', type=int, nargs=2, default=[384, 384],
                        help='Model input size as (height, width)')
    parser.add_argument('--crop_percent', type=float, nargs=4,
                        metavar=('left', 'right', 'top', 'bottom'),
                        default=[0.12, 0.08, 0.15, 0.10],
                        help='Crop percentages as left right top bottom')

    # Output configuration
    parser.add_argument('--output_base_path', type=str, default=None,
                        help='Base path for output features (auto-generated if not provided)')

    # Visualization options
    parser.add_argument('--visualize_features', action='store_true',
                        help='Enable feature map visualization')
    parser.add_argument('--no_visualize_features', dest='visualize_features',
                        action='store_false',
                        help='Disable feature map visualization')
    parser.add_argument('--max_visualize_frames', type=int, default=41182,
                        help='Maximum number of frames to visualize')
    parser.add_argument('--save_visualized_features', action='store_true',
                        help='Save visualized features as numpy array')

    # Set default value for visualize_features
    parser.set_defaults(visualize_features=True)

    return parser.parse_args()

def main():
    args = parse_args()

    # Set environment variables
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # Setup paths
    if args.output_base_path is None:
        video_path_last_path = os.path.dirname(args.video_path).split('/')[-1]
        args.output_base_path = f"./features_new_data/{args.feat_key}/{video_path_last_path}/{os.path.basename(args.video_path).split('.')[0]}"

    output_npy_path = f"{args.output_base_path}/video_features.npy"
    os.makedirs(args.output_base_path, exist_ok=True)

    # Convert crop percentages to dict format
    crop_percent = {
        'left': args.crop_percent[0],
        'right': args.crop_percent[1],
        'top': args.crop_percent[2],
        'bottom': args.crop_percent[3]
    }

    IMAGE_SIZE = tuple(args.image_size)

    # Initialize
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create visualization directory
    vis_dir = f"{args.output_base_path}/feature_visualizations"
    if args.visualize_features:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {args.video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps


    # Calculate sampling frame indices
    if total_frames < args.num_frames:
        indices = np.arange(total_frames)
        args.num_frames = total_frames
    else:
        indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=np.int32)
    
    # Load feature extraction model
    cfg_model = OmegaConf.load(args.model_config_path)['model']
    cfg_model = hydra.utils.instantiate(cfg_model)
    model = cfg_model.to(device).bfloat16()
    
    # Load model weights
    state_dict = load_file(args.ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()
    
    # Prepare lists to store features and frame info
    all_features = []
    frame_info = []
    visualized_features = []
    
    # Process video frames in batches
    processed_frames = 0
    for batch_start in tqdm(range(0, args.num_frames, args.batch_size),
                          desc="Processing video frames", position=0):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        batch_frames = []
        
        for frame_idx in batch_indices:
            # Set current frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert color space BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop frame
            cropped_frame = frame_rgb

            # Convert to PIL image and resize
            pil_img = Image.fromarray(cropped_frame).resize(IMAGE_SIZE)
            
            # Convert to tensor and normalize
            img_tensor = T.ToTensor()(pil_img).unsqueeze(0) * 2 - 1  # [-1, 1] range
            
            batch_frames.append(img_tensor)
            
            # Save frame information
            frame_info.append({
                'frame_idx': int(frame_idx),
                'timestamp': frame_idx / fps,
                'original_size': (width, height),
                'cropped_size': (cropped_frame.shape[1], cropped_frame.shape[0])
            })
        
        if not batch_frames:
            continue
            
        # Combine batch
        batch_tensor = torch.cat(batch_frames, dim=0).to(device).bfloat16()
        
        # Extract features
        with torch.no_grad():
            batch_features = model.get_features(batch_tensor, [args.caption] * len(batch_frames),
                                              t=None, feat_key=args.feat_key)
        
        # Process features for each frame
        for i in range(len(batch_frames)):
            # Get features for current frame
            feature = batch_features[i]
            
            # Visualize feature map
            if args.visualize_features and processed_frames < args.max_visualize_frames:
                feat_np = feature.detach().cpu().float().numpy()
                save_path = os.path.join(vis_dir, 
                                        f"frame_{frame_info[processed_frames]['frame_idx']:04d}_features.jpg")
                visualize_single_frame_features(
                    feat_np, 
                    save_path,
                    frame_info[processed_frames]['frame_idx'],
                    f"Feature Map ({feat_np.shape[0]}x{feat_np.shape[1]}x{feat_np.shape[2]})",
                    cmap='viridis',
                    dpi=150
                )
            
            # Flatten features: [C, H, W] -> [C*H*W] and use float16 to save space
            flat_features = feature.flatten().cpu().float().numpy().astype(np.float16)
            all_features.append(flat_features)
            
            processed_frames += 1
    
    # Release video resources
    cap.release()
    
    if not all_features:
        raise RuntimeError("Failed to extract features from any frame")
    
    # Merge all features
    all_features_array = np.vstack(all_features)
    feature_dim = all_features_array.shape[1]
    
    # Save features
    required_space = all_features_array.nbytes * 1.2  # Estimated required space
    if not check_disk_space(required_space, os.path.dirname(output_npy_path)):
        print("Warning: Insufficient disk space for saving features")
    else:
        np.save(output_npy_path, all_features_array)
    
    # Performance statistics
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_frame = total_time / args.num_frames
    
    print(f"Feature extraction completed!")
    print(f"Total frames processed: {len(all_features)}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per frame: {avg_time_per_frame:.4f}s")
    print(f"Output saved to: {output_npy_path}")

if __name__ == "__main__":
    main()
