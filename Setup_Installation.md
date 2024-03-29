# AI setup for Lissa (installation by Moufouli Bello)

### Basics  
``sudo apt update && sudo apt upgrade``  
``sudo apt install git``  
Sublime, install the GPG key:  
``wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -``  
Ensure apt is set up to work with https sources:  
``sudo apt-get install apt-transport-https``  
Select the channel to use:  
``echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list``  
``sudo apt-get install sublime-text``  

### Install Anaconda
get it: https://www.anaconda.com/products/individual#linux  
``sha256sum Anaconda3-2020.11-Linux-x86_64.sh``  
check key
``bash Anaconda3-2020.11-Linux-x86_64.sh``  
#### check which pip version is used (has to be the one inside the conda environment)
``which -a pip``

### Install general dependencies   
``sudo apt install g++``  
``conda install pytorch torchvision pydub simpleaudio torchaudio cudatoolkit=10.2 -c pytorch``
pip install simpleaudio
pip install pyddub
sudo apt-get install libasound2-dev

### Additional deps for sentence segmentation
conda install click==7.1.2
conda install -c conda-forge spacy

run ``python -m spacy download fr_core_news_sm``

### Setup systemd

copy lissa.service to ``/etc/systemd/system/lissa.service``
change path on line 5 to point to run.sh
```chmod +x run.sh```
systemctl daemon-reload
sudo systemctl enable lissa.service


### Install vosk STT
``git clone https://github.com/alphacep/vosk-api``  
``pip install vosk``  
``pip install sounddevice``  
``cd folder``  

use:  
``python test_microphone.py``  

list devices:  
``python test_microphone.py -l``  
change device for internal playback:  
``python test_microphone.py -d ...``  

get model: https://alphacephei.com/vosk/models
using vosk-model-fr-0.6-linto-2.2.0

### Install transformers
https://anaconda.org/conda-forge/transformers  
``conda install -c conda-forge transformers``  
``git clone https://huggingface.co/antoiloui/belgpt2``  

### Install tocatron TTS
``git clone https://github.com/Tomiinek/Multilingual_Text_to_Speech``  
?? ``git clone https://github.com/Tomiinek/WaveRNN``  
``pip install -q -U soundfile``  
``pip install -q -U phonemizer``  
``pip install -q -U epitran``  
(hint to check: pip list)  

use:  
``cd Multilingual_Text_to_Speech``  
``echo "1|Je vois une grande foule agitée.|00-de|de*0.2:fr*0.8" python /home/panorama/tts/Multilingual_Text_to_Speech/synthesize.py --checkpoint /home/panorama/tts/checkpoints/generated_switching.py``  

tts other possiblities: svoxpico/pico2wave, espak with mrolla, python gtts (google)

### sound stuff  
http://manpages.ubuntu.com/manpages/trusty/man1/aplay.1.html  
??force rate to change playback speed and pitch:  
-r, --rate=#<Hz>  
  
http://www.mplayerhq.hu/DOCS/HTML/en/index.html  
mplayer -speed

### handy stuff
``which -a pip``  
``conda list``  
``pip list``  
``python -m sounddevices``  


??try to install as much as possible in conda  
??conda install juptyer, usw..  
