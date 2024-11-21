import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    # parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--filename', type=str, default='camera_record')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
  
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print('Frame rate is:', frame_rate)
    
    if args.pred_only: 
        output_width = frame_width
    else: 
        output_width = frame_width * 2 + margin_width
    output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.filename))[0] + '.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
    
    while True:
        # 读取摄像头的图像帧
        ret, frame = cap.read()
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        image = frame.copy()
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not ret:
            break
            
        depth = depth_anything.infer_image(frame, args.input_size)
    
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imshow('Depth Pridiction', depth)
            out.write(depth)
        else:
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([frame, split_region, depth])
            cv2.imshow('Depth Pridiction', combined_frame)
            out.write(combined_frame)
    
        # 检测按下键盘上的q键退出循环
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # 释放摄像头，关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    
    # for k, filename in enumerate(filenames):
    #     print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
    #     raw_video = cv2.VideoCapture(filename)
    #     frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
    #     if args.pred_only: 
    #         output_width = frame_width
    #     else: 
    #         output_width = frame_width * 2 + margin_width
        
    #     output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
    #     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
    #     while raw_video.isOpened():
    #         ret, raw_frame = raw_video.read()
    #         if not ret:
    #             break
            
    #         depth = depth_anything.infer_image(raw_frame, args.input_size)
            
    #         depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    #         depth = depth.astype(np.uint8)
            
    #         if args.grayscale:
    #             depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    #         else:
    #             depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
    #         if args.pred_only:
    #             out.write(depth)
    #         else:
    #             split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
    #             combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                
    #             out.write(combined_frame)
        
    #     raw_video.release()
    #     out.release()
