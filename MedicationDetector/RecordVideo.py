# Record the video of a person taking medicine from a fixed camera, which is done by the ipCam app
# Input: filePath1 - IPcam access path, which includes IP address and access video name
#        outputDir - the path of output video
#        outputFname - the name of output video
# Output: a mp4 video will be generated in the specifiied path, fps 12.5, size 256x340

import os
import argparse

import cv2
import mediapipe as mp


args = argparse.ArgumentParser()
args.add_argument("--mode", default='video')
# The ipCam video is accessed by video.mjpg
args.add_argument("--filePath1", default="http://192.168.12.184:8080/video.mjpg")
args.add_argument("--outputDir", default="./MedicationDetector/videos")
args.add_argument("--outputFname", default="a1.mp4")

args = args.parse_args()


outputDir = args.outputDir
if not os.path.exists(outputDir):
    os.makedirs(outputDir)


cap1 = cv2.VideoCapture(args.filePath1)
ret1, frame1 = cap1.read()
print(str(frame1.shape[1]) + " " + str(frame1.shape[0]))
output_video1 = cv2.VideoWriter(os.path.join(outputDir, args.outputFname),
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               12.5,
                               (256, 340))
                               # 12.5 is the fps set in ipCam
                               # frame1 shapse was 480 * 360, but TimeSformer requires resize short edge size to be 256

while ret1:

    cv2.imshow('frame1', frame1)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cap1.release()
        output_video1.release()
        cv2.destroyAllWindows()
        break
    # frame1 shapse is 480 * 360, but TimeSformer requires resize short edge size to be 256
    resize1 = cv2.resize(frame1, (256, 340), interpolation = cv2.INTER_AREA)
    output_video1.write(resize1)

    ret1, frame1 = cap1.read()
    
cv2.destroyAllWindows()

