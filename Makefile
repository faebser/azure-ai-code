.PHONY: setup

setup:
	mkdir -p checkpoints/
	mkdir -p recordings/
	git clone https://github.com/Tomiinek/Multilingual_Text_to_Speech
	cd checkpoints/ && curl -O -L "https://github.com/Tomiinek/Multilingual_Text_to_Speech/releases/download/v1.0/generated_switching.pyt"
	cd checkpoints/ && scp -r datascience@40.127.190.201:/home/datascience/notebooks/moufouli/results-finetune/checkpoint-2500 .
	cd checkpoints/ && curl -O -L "https://alphacephei.com/vosk/models/vosk-model-fr-0.6-linto-2.0.0.zip" && unzip "vosk-model-fr-0.6-linto-2.0.0.zip" -d vosk-model
