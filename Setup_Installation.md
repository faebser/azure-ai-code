# AI setup for Lissa (installation by Moufouli Bello)

``sudo apt update && sudo apt upgrade``

### helpful

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
#### check which pip version is used
``which -a pip``

### Install dependencies   
``sudo apt install g++``  
``conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch``

#### Install vosk STT
``git clone https://github.com/alphacep/vosk-api``
``pip install vosk``
``pip install sounddevice``
``cd folder``
``python test_microphone.py``

#### Install transformers
https://anaconda.org/conda-forge/transformers  
``conda install -c conda-forge transformers``  
``git clone https://huggingface.co/antoiloui/belgpt2``  

#### Install tocatron TTS
``git clone https://github.com/Tomiinek/Multilingual_Text_to_Speech``  
``git clone https://github.com/Tomiinek/WaveRNN``  

``cd Multilingual_Text_to_Speech``  
``pip install -r requirements.txt``  
(hint to check: pip list)

??try one by one in conda  
??conda install juptyer, usw..  
