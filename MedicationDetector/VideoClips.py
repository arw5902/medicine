# Divide a video into clips, need to set clipFrameNum to change number of frames
# Input: filePath - video file name to be clipped
#        video_dir - the dir of video, which is not supposed to change
#        clipDir - the directory of output video clips
#        clipFrameNum - how many frames in one clip
# Output: video clips with specified frame number will be generated in the specifiied clip dir

import os
import argparse
import cv2

args = argparse.ArgumentParser()
args.add_argument("--filePath", default='t1_.mp4')
args.add_argument("--clipDir", default='./MedicationDetector/clip_a_75')
args.add_argument("--clipFrameNum", default=75)
args = args.parse_args()
video_dir = './MedicationDetector/videos' # video_dir should already exist
clipDir = args.clipDir

if not os.path.exists(clipDir):
    os.makedirs(clipDir)
fps = 12.5 # 12.5 is the fps set in VideoWriter when generating the videos

filePath = os.path.join(video_dir, args.filePath)

count = 1
cap = cv2.VideoCapture(filePath)
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
 
# experiment for cropping at center
mid_x, mid_y = int(224/2), int(296/2)
cw2, ch2 = int(224/2), int(224/2) 

while True:
    ret, frame = cap.read()
    print(str(frame.shape[0]) + " " + str(frame.shape[1]))
    if not ret:
        break
    output_clip = cv2.VideoWriter(os.path.join(
        clipDir, args.filePath.rsplit( ".", 1 )[ 0 ] + 'clip' + str(count) + '.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (224, 296)) # with short edge size set to 224, the random_short_side_scale_jitter will not scale the image    
    resize1 = cv2.resize(frame, (224, 296)) 
    output_clip.write(resize1)
    
    for i in range(1, args.clipFrameNum): #starting from the next frame since the first one is already written.
        ret, frame = cap.read()
        if not ret:
            break
        resize1 = cv2.resize(frame, (224, 296))
        output_clip.write(resize1)

    cap.release()
    output_clip.release()
    
    # open again to get ready for the next clip
    cap = cv2.VideoCapture(filePath)
    for i in range(0, count):
        ret, frame = cap.read()
        if not ret:
            break

    count += 1
    if count + args.clipFrameNum > totalFrames:
        break

cap.release()
