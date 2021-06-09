#!/usr/bin/env python3

#

import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
from pydub import AudioSegment
from pydub.playback import play
import datetime
import json
import traceback

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


# Build models
current_dir = os.getcwd()

vosk_model = vosk.Model(os.path.join(current_dir, "checkpoints", "vosk-model"))
gpt2_model = GPT2LMHeadModel.from_pretrained("checkpoints/checkpoint-2500/")
tokenizer = GPT2TokenizerFast.from_pretrained("antoiloui/belgpt2", model_max_length=768, pad_token='<|pad|>')

# put gpt2 on gpu
device = torch.device('cuda')
gpt2_model.cuda()
# put gpt2 in eval mode
gpt2_model.eval()

# import tacotron stuff
tacotron_dir = "Multilingual_Text_to_Speech"
tacotron_chpt = "generated_switching.pyt"

#os.chdir(os.path.join(current_dir, tacotron_dir))
sys.path.append(os.path.join('/home/panorama/azure-ai-code/', tacotron_dir))

if "utils" in sys.modules: del sys.modules["utils"]

#print(os.getcwd())

from synthesize import synthesize
from params.params import Params as hp
from utils import build_model, audio

os.chdir(current_dir)

model_taco = build_model(os.path.join(current_dir, "checkpoints", tacotron_chpt))
model_taco.eval()
model_taco.cuda()

##
# HELPERS
##
##

def date_name():
    a = datetime.datetime.now()
    return "{}_{}_{}-{}_{}_{}".format(a.year, a.month, a.day, a.hour, a.minute, a.second)


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def generate_text(question):
    # TODO add hidden prompt
    prompt = "{}A: {}\n B: ".format(tokenizer.bos_token, question)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    to_return = []

    output = gpt2_model.generate(
                inputs,
                do_sample=True,
                top_k=50,
                max_length=len(prompt) + 100,
                top_p=0.65,
                num_return_sequences=2,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.65
    )

    # Decode it
    decoded_output = []
    for sample in output:
        decoded_output.append(tokenizer.decode(sample, skip_special_tokens=True))
    #print(decoded_output)
    for o in decoded_output:
        b = o.split( 'B:' )[1]
        print(b)
        to_return.append(b + "|00-de|fr")

    # sort by length and return only the longest
    r = sorted(to_return, key=len)

    # TODO save generated answer

    return r[-1]

def generate_audio(answer):
    spectogram = synthesize(model_taco, "|" + answer)
    return audio.inverse_spectrogram(spectogram, not hp.predict_linear)

def play_audio(speech_audio):
    audio.save(speech_audio, os.path.join('recordings', date_name()) +'.wav')
    #talk = AudioSegment.from_raw(audio)
    #play(talk)
    #talk.export(os.path.join('recordings', date_name) + '.mp3', format=mp3)
##
# CONFIG
##

DEVICE_ID =  0
# get device from vosk example
BLOCK_SIZE = 80000
SAMPLE_RATE = None # set to None for auto samplerate

if SAMPLE_RATE is None:
    device_info = sd.query_devices(DEVICE_ID, 'input')
    # soundfile expects an int, sounddevice provides a float:
    SAMPLE_RATE = int(device_info['default_samplerate'])

q = queue.Queue()

try:
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize = BLOCK_SIZE, device=DEVICE_ID,
                           dtype='int16', channels=1, callback=callback):

            rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    #r = rec.Result()
                    r = json.loads(rec.Result())
                    print("result: {}".format(r['text']))
                    answer = generate_text(r['text'])
                    _audio = generate_audio(answer)
                    play_audio(_audio)
                    with q.mutex: q = queue.Queue()
                else:
                    pass
                    #print(rec.PartialResult())

except KeyboardInterrupt:
    print('\nDone')
except Exception as e:
    traceback.print_tb(e.__traceback__)
    print(type(e).__name__ + ': ' + str(e))
    exit(-1)
