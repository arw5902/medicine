#!/bin/bash

min=41
max=44
#for i in {1..$max}; do
for ((i=$min; i<=$max; i++)); do
    echo $(printf "t%d_.mp4" $i)
    python3 VideoClips.py --filePath $(printf "t%d_.mp4" $i)
    #python3 VideoClips_8frames.py --filePath $(printf "a%d.mp4" $i)
done
