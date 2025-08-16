# Label each clip of a holdout test video according to ground truth csv file
# specified by args.filePath1
# Input: basePath - base directory of the videos and clips
#        filePath1 - ground truth "vsamples_fullvideo.csv" by default, which specifies the frame number contains the medicine taking moment
#        videoName - file name of the holdout test video
#        FrameInterval - the frame intervals between frames next to each other 
#        clipSize - NO. of frames in each clip
# Output: csamples_live_{holdout test video name}.csv - a csv file labeling each clip: 0 - no taking; 1 - taking 

import os
import argparse
import pandas as pd
import cv2
import csv
import math

args = argparse.ArgumentParser()
args.add_argument("--basePath", default='./')
args.add_argument("--filePath1", default='vsamples_fullvideo.csv')
args.add_argument("--FrameInterval", default=5)  # was 10 originally
args.add_argument("--videoName", default='t22')
args.add_argument("--clipSize", default=100)
args = args.parse_args()

csv_dir = os.path.join(args.basePath) + 'csv/holdout/'
if args.clipSize == 75:
    clip_frame_num = 75
    liveclip_dir = csv_dir + 'frames_75'
    clipfile_dir = os.path.join(args.basePath, 'clip_a_75')
elif args.clipSize == 100:
    clip_frame_num = 100
    liveclip_dir = csv_dir + 'frames_100'
    clipfile_dir = os.path.join(args.basePath, 'clip_a_100')
elif args.clipSize == 125:
    clip_frame_num = 125
    liveclip_dir = csv_dir + 'frames_125'
    clipfile_dir = os.path.join(args.basePath, 'clip_a_125')

#afterwardAdjust = math.floor(clip_frame_num/2) - 5 # -5 to make the 1's sample 10% of total
afterwardAdjust = 0 #set to 0 to ignore this option for now

filePath1 = os.path.join(csv_dir, args.filePath1)

live_csv = os.path.join(liveclip_dir, 'csamples_live_'+args.videoName+'.csv')
#print(live_csv)
if os.path.exists(live_csv):    # this file should already be created by inf.py
    videoSampleF = open(filePath1, "r")
    #print(videoSampleF)
    videoSample = csv.reader(videoSampleF)
    #clipSampleF = open(live_csv, "w", encoding='utf-8')
    clipSampleDF = pd.read_csv(live_csv, sep=',')
    
    labels = []
    # Although for loop is used, only single video args.videoName is processed.
    # This can be modified to handle all videos in args.filePath1
    for col in videoSample:
        fname = col[0]
        frameNumber = col[1]
        if fname and frameNumber:
            print(fname, " ", frameNumber)
            if fname == args.videoName + "_":
                count = 1
                while True:
                    fileName = os.path.join(clipfile_dir, fname + "clip" + str(count) + ".mp4")
                    print(fileName)
                    if os.path.exists(fileName):
                        if count >= int(frameNumber) - int(clip_frame_num) and count <= int(frameNumber) - afterwardAdjust:
                            labels.append(1)
                        else:
                            labels.append(0)
                    else:
                        break
                    count += args.FrameInterval
                break
    
    print(labels)
    clipSampleDF['Label'] = pd.Series(labels)
    clipSampleDF.to_csv(live_csv, index=False)
    
    videoSampleF.close()
