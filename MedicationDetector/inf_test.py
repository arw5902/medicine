#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate the csv file for both ground truth and model prediction, this is to 
# compare the two results side by side. This requires the test dataset csv file
# placed in the csv/{frame_number}/ folder, and the test dataset clips placed
# in the location specified in the csv file.
# Input: basePath - base directory of the videos and clips
#        clipSize - NO. of frames in each clip
# Output: generate the csv file that shows the ground truth of test.csv (15% of
#         all clips sent to TimeSformer) and the model prediction side by side.
#         To be more specific, for each clip file in test.csv, it checks for 
#         the medication behavior and record the Confidence percentage in the
#         resulting test_ret.csv.

import os
import argparse
import pandas as pd
from datetime import datetime
import shutil
import math
import torch
import av
import cv2
import numpy as np
from timesformer.datasets.decoder import decode
from timesformer.datasets.utils import tensor_normalize, spatial_sampling
from timesformer.models.vit import TimeSformer
import timesformer.utils.logging as logging
logger = logging.get_logger(__name__)

args = argparse.ArgumentParser()
args.add_argument("--basePath", default='./')
args.add_argument("--clipSize", default=75)
args = args.parse_args()

n_frames = 32 # sampling frame number
video_fps = 12.5 # was 12.5

if args.clipSize == 75:
    #clip_dir = 'div_clips_75'
    sample_dir = 'samples_75'
    checkpoint_file = './model/75/checkpoint_epoch_00014.pyth'
    #clip_frame_num = 75
    testClipFile = 'csv/75/test_75.csv'
    retClipFile = 'csv/75/test_75_ret.csv'
    s_rate = 2
    t_fps = 11
elif args.clipSize == 100:
    #clip_dir = 'div_clips_100'
    sample_dir = 'samples_100'
    checkpoint_file = './model/100/checkpoint_epoch_00015.pyth' # was checkpoint_epoch_00036.pyth for SMC paper
    #clip_frame_num = 100
    testClipFile = 'csv/100/test_100.csv'
    retClipFile = 'csv/100/test_100_ret.csv'
    s_rate = 3
    t_fps = 12
elif args.clipSize == 125: 
    #clip_dir = 'div_clips_125'
    sample_dir = 'samples_125'
    checkpoint_file = './model/125/checkpoint_epoch_00026.pyth'
    #clip_frame_num = 125
    testClipFile = 'csv/125/test_125.csv'
    retClipFile = 'csv/125/test_125_ret.csv'
    s_rate = 4
    t_fps = 13

model = TimeSformer(img_size=224, num_classes=2, num_frames=n_frames, attention_type='divided_space_time',  pretrained_model=checkpoint_file)
frame_num_limit = 1000    #limit for video feed

def sig(x):
 return 1/(1 + np.exp(-x))

@torch.no_grad()
def detect_a_sample(c_file, s_file, m):
    video_container = av.open(c_file)
    frames = decode(
        video_container,
        sampling_rate=s_rate, #cfg.DATA.SAMPLING_RATE:frame sampling rate (interval between two sampled frames).
        num_frames=n_frames,
        clip_idx=0, #perform random temporal sampling
        num_clips=1, #overall number of clips to uniformly sample from the given video for testing
        video_meta={}, #self._video_meta[index],
        target_fps=t_fps, #cfg.DATA.TARGET_FPS, set to 10 to match with the 75 clip size
        backend="pyav", #cfg.DATA.DECODING_BACKEND,
        max_spatial_scale=224, #keep the aspect ratio and resize the frame so that the shorter edge size
                               #is max_spatial_scale. Only used in `torchvision` backend.
        )
    print(frames.shape)
    
    #write the sampled frames to a video file
    output_clip = cv2.VideoWriter(s_file,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps,
        (224, 296))
  
    for i in range(1, n_frames):
        output_clip.write(frames.numpy())
    output_clip.release()
    video_container.close()
    
    # Perform color normalization.
    frames = tensor_normalize(
        frames,
        [0.45, 0.45, 0.45], #cfg.DATA.MEAN
        [0.225, 0.225, 0.225] #cfg.DATA.STD
    )
    # T H W C -> C T H W.
    frames = frames.permute(3, 0, 1, 2)
    # Perform data augmentation.
    frames = spatial_sampling(
        frames,
        spatial_idx=1, #spatial_sample_index, if -1, perform random spatial sampling. If 0, 1,
                        #or 2, perform left, center, right crop if width is larger than
                        #height, and perform top, center, buttom crop if height is larger
                        #than width.
        min_scale=224,
        max_scale=224,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False
    )
    # Perform temporal sampling from the fast pathway.
    frames = torch.index_select(
         frames,
         1,
         torch.linspace(
             0, frames.shape[1] - 1, n_frames #cfg.DATA.NUM_FRAMES
         ).long(),
    )
    

    frames=torch.unsqueeze(frames, 0) # make batch size as 1
    #print(frames.shape)
    
    m.eval()
    pred = m(frames,)
    logger.info("In topks_correct...pred is {}".format(pred))
    m_norm = torch.nn.Softmax(dim=1)
    pred_norm = m_norm(pred)
    # Find the top max_k(=1) predictions for each sample
    _top_max_1_vals, top_max_1_inds = torch.topk(
        # sig(pred), 1, dim=1, largest=True, sorted=True
        pred_norm, 1, dim=1, largest=True, sorted=True
    )
    logger.info("_top_max_1_vals is {}, top_max_1_inds is {}".format(_top_max_1_vals, top_max_1_inds))
    # val_1 = float("{:.4f}".format(sig(pred[0][1].item())))
    # val_0 = float("{:.4f}".format(sig(pred[0][0].item())))
    val_1 = float("{:.4f}".format(pred_norm[0][1].item()))
    val_0 = float("{:.4f}".format(pred_norm[0][0].item()))
    print(val_0, val_1)
    if top_max_1_inds == 1:
        return 1, val_1, val_0
    else:
        return 0, val_1, val_0

def main():
    logging.setup_logging(".")
    takens = 0
    csv_file = os.path.join(args.basePath, testClipFile)
    if not os.path.exists(csv_file):
        print("test csv file does not exist")
        return
    df = pd.read_csv(csv_file, sep=',', skiprows=1, names=["FilePath", "Label"]) #, "Predict", "Confidence", "Confidence_0"])
    predict = []
    conf = []
    conf_0 = []
    for f in df.FilePath:
        print(f)
        sample_file = os.path.join(
            args.basePath, sample_dir, os.path.basename(f))
        print(sample_file)
        ret_d, val_1, val_0 = detect_a_sample(f, sample_file, model)
        predict.append(ret_d)
        conf.append(val_1)
        conf_0.append(val_0)
    
        if (ret_d):
            print(os.path.basename(f) + " Medicine was taken")
            
            takens += 1
        else:
            print(os.path.basename(f) + " Medicine was not taken")
            if takens > 0:
                takens -= 1
    
    df["Predict"] = pd.Series(predict)  
    df["Confidence"] = pd.Series(conf)
    df["Confidence_0"] = pd.Series(conf_0)  
    df.to_csv(os.path.join(args.basePath, retClipFile), index=False)
    print(predict)
    print(conf)
    print(conf_0)
           
    
    return (str(takens))
    
if __name__ == "__main__":
    main()
