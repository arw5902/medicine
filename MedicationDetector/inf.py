#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Live feed medication behavior detection for one video, generate csamples_live_a##.csv
# Input: livePath - video feed file name
#        videoDir - dir to store videos
#        clipSize - NO. of frames in each clip
#        clipDir - dir to store clips
#        sampleDir - dir to store sample clips
#        videoName - video to be tested
# Output: print whether a medication behavior has been detected, if yes then
#         generate a proof video. 

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

from queue import Queue
#from deepface import DeepFace

args = argparse.ArgumentParser()
args.add_argument("--basePath", default='./')
args.add_argument("--livePath", default="http://192.168.12.193:8080/video.mjpg")
args.add_argument("--videoDir", default='videos')
args.add_argument("--clipSize", default=75)
args.add_argument("--clipDir", default='div_clips_75')
args.add_argument("--sampleDir", default='samples_75')
args.add_argument("--videoName", default='t1')
args = args.parse_args()

video_file = os.path.join(args.basePath, args.videoDir, args.videoName+'_.mp4')
record_fname = 'csv/MedicationRecord.csv'
video_fps = 12.5
n_frames = 32 # sampling frame number

if args.clipSize == 75:
    clip_dir = 'div_clips_75'
    cample_dir = 'samples_75'
    frame_gap = 5 # for 75 frames clip, we are going to didscard half clips
    liveClipFile = 'csv/frames_75/csamples_live_'+args.videoName+'.csv'
    checkpoint_file = './MedicationDetector/csv/20241110/checkpoint_epoch_00014.pyth'
    s_rate = 2
    t_fps = 11
    clip_frame_num = 75
elif args.clipSize == 100:
    clip_dir = 'div_clips_100'
    cample_dir = 'samples_100'
    frame_gap = 10    
    od_fname = 'csv/OverdoseRecord.csv'
    liveClipFile = 'csv/frames_100/csamples_live_'+args.videoName+'.csv'
    checkpoint_file = './MedicationDetector/csv/20241109/checkpoint_epoch_00036.pyth'
    s_rate = 3
    t_fps = 12
    clip_frame_num = 100
elif args.clipSize == 125:  
    clip_dir = 'div_clips_125'
    cample_dir = 'samples_125'
    frame_gap = 10
    liveClipFile = 'csv/frames_125/csamples_live_'+args.videoName+'.csv'
    checkpoint_file = './MedicationDetector/csv/20241111/checkpoint_epoch_00026.pyth'
    s_rate = 4
    t_fps = 13
    clip_frame_num = 125

model = TimeSformer(img_size=224, num_classes=2, num_frames=n_frames, attention_type='divided_space_time',  pretrained_model=checkpoint_file)
frame_num_limit = 1000    #limit for live video feed

def sig(x):
 return 1/(1 + np.exp(-x))

@torch.no_grad()
def detect_a_sample(c_file, s_file, m):
    video_container = av.open(c_file)
    #dummy_video = torch.randn(2, 3, 4) #(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)
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

def detect_face(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    if out.detections is not None:
        print("face detected")
        return True
    else:
        #print("face not detected")
        return False


def main():
    logging.setup_logging(".")
    takens = 0
    
    cap = cv2.VideoCapture(video_file)
    count = 1
    queue_size = int(math.ceil(clip_frame_num / frame_gap))
    clip_fname_q = Queue()
    output_clip_q = Queue()
    discard = False # toggle delete clip file or not for 75-frame clips
    
    while True: #False:
        ret, frame = cap.read()
        #print(str(frame.shape[1]) + " " + str(frame.shape[0]))
        if not ret:
            break

        if count % frame_gap == 1:
            if clip_fname_q.qsize() >= queue_size:
                a_cfname = clip_fname_q.get()
                a_clip = output_clip_q.get()
                a_clip.release()
                # A video clip has been generated, send it to the TimeSformer model
                sample_file = os.path.join(
                    base_path, args.sampleDir, os.path.basename(a_cfname))
                if (args.clipSize == 75 and discard):
                    os.remove(a_cfname)
                else:
                    ret_d, val_1, val_0 = detect_a_sample(a_cfname, sample_file, model)
                    live_csv = os.path.join(base_path, liveClipFile)
                    if not os.path.exists(live_csv):
                        # Create an empty DataFrame if the file doesn't exist
                        df1 = pd.DataFrame()
                        df1.to_csv(live_csv, index=False)  # Save the empty DataFrame to create the file
                        df1 = pd.read_csv(live_csv, sep=',', names=["FilePath", "Label", "Predict", "Confidence", "Confidence_0"])
                    else:
                        df1 = pd.read_csv(live_csv, sep=',')
                    new_row = {"FilePath":a_cfname, "Predict":ret_d, "Confidence":val_1, "Confidence_0":val_0}
                    df1 = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)
                    df1.to_csv(live_csv, sep=',', index=False)
                    if (ret_d):
                        print(os.path.basename(a_cfname) + " Medicine was taken")
                        
                        takens += 1
                        if takens == 3:
                            print("Detection succeeded.")
                            record_file = os.path.join(base_path, record_fname)
                            saved_file = os.path.join(base_path, "saved", os.path.basename(a_cfname))
                            os.system("ffmpeg -i "+a_cfname+" -filter:v 'crop=320:200:20:20' -vcodec h264 "+saved_file)
                            shutil.copyfile(a_cfname, saved_file)
                            df = pd.read_csv(record_file, sep=',')
                            new_row = {"Date": datetime.today().strftime('%m/%d/%Y'), "Time": datetime.today().strftime('%H:%M:%S'), "Clip": saved_file}
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            df.to_csv(record_file, sep=',', index=False)
                            break
                    else:
                        print(os.path.basename(a_cfname) + " Medicine was not taken")
                        if takens > 0:
                            takens -= 1
                discard = not discard
                
                    
            clip_fname = os.path.join(
                base_path, args.clipDir, args.videoName + '_clip' + str(count) + '.mp4')
            clip_fname_q.put(clip_fname)
            # generate the clip file for reference purpose
            output_clip = cv2.VideoWriter(clip_fname,
                cv2.VideoWriter_fourcc(*'mp4v'),
                video_fps,
                (224, 296))
            output_clip_q.put(output_clip)
        resize1 = cv2.resize(frame, (224, 296))
        for i in output_clip_q.queue:
            i.write(resize1)
                    
        count += 1
        if count + clip_frame_num > frame_num_limit:
            break
    
    return (str(takens))
    
if __name__ == "__main__":
    main()
