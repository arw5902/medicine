#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Live feed medication behavior detection for one video, generate csamples_live_a##.csv
# Input: livePath - video feed file name
#        videoDir - dir to store videos
#        clipDir - dir to store clips
#        sampleDir - dir to store sample clips
#        videoName - video to be tested
# Output: print whether a medication behavior has been detected, if yes then
#         generate a proof video.
#         100-frame clip size is chosen to be the most reliable. This size is 
#         used in live detection test; medication taking time is recorded in
#         MedicationRecord.csv. Overdose record is recorded in OverdoseRecord.csv.

import os
import argparse
import mediapipe as mp
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
args.add_argument("--livePath", default="http://192.168.12.184:8080/video.mjpg")
args.add_argument("--videoDir", default='videos')
args.add_argument("--clipDir", default='div_clips_100') # clips divided by this code
args.add_argument("--sampleDir", default='samples_100')
args.add_argument("--videoName", default='live')
args = args.parse_args()

n_frames = 32 # sampling frame number
frame_gap = 10 # for 100 frames clip
video_fps = 12.5
base_path = './MedicationDetector/'
record_fname = 'csv/MedicationRecord.csv'
od_fname = 'csv/OverdoseRecord.csv'
liveClipFile = 'csv/frames_100/csamples_live_'+args.videoName+'.csv'

checkpoint_file = './MedicationDetector/csv/20241109/checkpoint_epoch_00036.pyth'
model = TimeSformer(img_size=224, num_classes=2, num_frames=n_frames, attention_type='divided_space_time',  pretrained_model=checkpoint_file)
frame_num_limit = 300    #limit for live video feed

def sig(x):
 return 1/(1 + np.exp(-x))

@torch.no_grad()
def detect_a_sample(c_file, s_file, m):
    video_container = av.open(c_file)
    frames = decode(
        video_container,
        sampling_rate=3, #cfg.DATA.SAMPLING_RATE:frame sampling rate (interval between two sampled frames).
        num_frames=n_frames,
        clip_idx=0, #perform random temporal sampling
        num_clips=1, #overall number of clips to uniformly sample from the given video for testing
        video_meta={}, #self._video_meta[index],
        target_fps=12, #cfg.DATA.TARGET_FPS, set to 10 to match with the 75 clip size
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
    #val = float("{:.4f}".format(_top_max_1_vals.item()))
    logger.info("_top_max_1_vals is {}, top_max_1_inds is {}".format(_top_max_1_vals, top_max_1_inds))
    # val_1 = float("{:.4f}".format(sig(pred[0][1].item())))
    # val_0 = float("{:.4f}".format(sig(pred[0][0].item())))
    val_1 = float("{:.4f}".format(pred_norm[0][1].item()))
    val_0 = float("{:.4f}".format(pred_norm[0][0].item()))
    print(val_0, val_1)
    if top_max_1_inds == 1:
    #if val_1 >= th_100:
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
    clip_frame_num = 100
    recorded = 0
    takens = 0
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence=0.5) # for short range detection
    # Video capturing
    cap = cv2.VideoCapture(args.livePath)
    args.videoName = datetime.today().strftime('%Y-%m-%d-%H-%M-%S_') + args.videoName
    video_file = os.path.join(base_path, args.videoDir, args.videoName+'_.mp4')
    output_video = cv2.VideoWriter(video_file,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps,
        (224, 296))
    
    # Flag to track if the face is detected
    face_detected = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i in range(30):
            # Convert the fqqqrame to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
        
            # Draw detections
            if results.detections:
                face_detected = True
                print("face detected:", i)
                break
            else:
                face_detected = False

            ret, frame = cap.read()
            if not ret:
                break
    
        # Start video recording when a face is detected
        if face_detected == True:
            print("starts writing")
            recorded = 1
            for i in range(200):
                resized = cv2.resize(frame, (224, 296), interpolation = cv2.INTER_AREA)
                output_video.write(resized)
                ret, frame = cap.read()
                if not ret:
                    break
            face_detected = False
        else:
            # we only get one video for now
            print("no face")
            if recorded == 1:
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release resources
    cap.release()
    output_video.release()

    #return takens
    
    cap = cv2.VideoCapture(video_file)
    count = 1
    queue_size = int(math.ceil(clip_frame_num / frame_gap))
    clip_fname_q = Queue()
    output_clip_q = Queue()
     
    while True: #False:
        ret, frame = cap.read()
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
                    if takens >= 3:
                        print("Detection succeeded.")
                        record_file = os.path.join(base_path, record_fname)
                        saved_file = os.path.join(base_path, "saved", os.path.basename(a_cfname))
                        #os.system("ffmpeg -i "+a_cfname+" -filter:v 'crop=320:200:20:20' -vcodec h264 "+saved_file)
                        #os.system("ffmpeg -i "+a_cfname+" -vcodec h264 "+saved_file)
                        #shutil.copyfile(a_cfname, saved_file)
                        df = pd.read_csv(record_file, sep=',')
                        new_row = {"Date": datetime.today().strftime('%m/%d/%Y'), "Time": datetime.today().strftime('%H:%M:%S'), "Clip": saved_file}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        df.to_csv(record_file, sep=',', index=False)
                        # check for overdose
                        result = df[df["Date"] == datetime.today().strftime('%m/%d/%Y')]
                        print(result)
                        if (len(result) > 3):
                            od_file = os.path.join(base_path, od_fname)
                            df_od = pd.read_csv(od_file, header=None, usecols=[0])
                            new_od_row = {datetime.today().strftime('%m/%d/%Y')}
                            df_od = pd.concat([df_od, pd.DataFrame([new_od_row])], ignore_index=True)
                            df_od.to_csv(od_file, header=None, index=False)
                        break
                else:
                    print(os.path.basename(a_cfname) + " Medicine was not taken")
                    if takens > 0:
                        takens -= 1                
                    
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
