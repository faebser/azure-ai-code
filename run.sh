#!/bin/bash
rm /home/panorama/azure-ai-code/recordings/*.wav -f
source  /home/panorama/anaconda3/etc/profile.d/conda.sh
pulseaudio --start
conda activate base
python /home/panorama/vosk-api/python/example/test_microphone.py -l
cd /home/panorama/azure-ai-code/
python installation.py
