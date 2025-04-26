#!/bin/bash

min=41
max=44
#for i in {1..$max}; do
for ((i=$min; i<=$max; i++)); do
    echo $(printf "t%d.mp4" $i)
    #python3 VideoClips.py --filePath $(printf "a%d.mp4" $i)
    #python3 VideoClips_8frames.py --filePath $(printf "a%d.mp4" $i)
    #python3 inf_75.py --videoName $(printf "t%d" $i)
    #python3 inf_100.py --videoName $(printf "t%d" $i)
    #python3 inf_125.py --videoName $(printf "t%d" $i)
    python3 ClipLabel_live.py --videoName $(printf "t%d" $i)
done
