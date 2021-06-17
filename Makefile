.PHONY: setup

setup:
	mkdir -p checkpoints/
	mkdir -p recordings/
	git clone https://github.com/Tomiinek/Multilingual_Text_to_Speech
	cd checkpoints/ && curl -O -L "https://github.com/Tomiinek/Multilingual_Text_to_Speech/releases/download/v1.0/generated_switching.pyt"
	cd checkpoints/ && curl -O -L "https://alphacephei.com/vosk/models/vosk-model-fr-0.6-linto-2.0.0.zip" && unzip "vosk-model-fr-0.6-linto-2.0.0.zip" -d vosk-model

save:
	cd checkpoints/ && scp -r datascience@137.135.162.192:/home/datascience/notebooks/moufouli/results/checkpoint-3500 .

v2:
	cd checkpoints/ && scp -r datascience@13.74.133.178:/home/datascience/notebooks/azure-ai-code/results3-finetune/checkpoint-2000 v2/

v3:
	cd checkpoints/ && scp -r datascience@13.74.133.178:/home/datascience/notebooks/azure-ai-code/results-combined v3/

v4:
	cd checkpoints/ && scp -r datascience@52.164.248.92:/home/datascience/notebooks/azure-ai-code/results_v4/checkpoint-3250 v4/
