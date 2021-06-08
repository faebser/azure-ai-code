#!/usr/bin/env python3

#

import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
import simpleaudio

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Build models

vosk_model = vosk.Model("vosk_model")
gpt2_model = GPT2LMHeadModel.from_pretrained("checkpoints/checkpoint-2500/")
tokenizer = GPT2TokenizerFast.from_pretrained("antoiloui/belgpt2", model_max_length=768, pad_token='<|pad|>')

# put gpt2 on gpu
device = torch.device('cuda')
gpt2_model.cuda()
# put gpt2 in eval mode
gpt2_model.eval()

# import tacotron stuff
current_dir = os.getcwd()
tacotron_dir = "Multilingual_Text_to_Speech"
tacotron_chpt = "generated_switching.pyt"

os.chdir(os.path.join(current_dir, tacotron_dir))

if "utils" in sys.modules: del sys.modules["utils"]

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

    output = model_test.generate(
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

    return r[-1]

def generate_audio(answer):
    spectogram = synthesize(model_taco, "|" + answer)
    return audio.inverse_spectrogram(a, not hp.predict_linear)

def play_audio(audio):
    print("please implement me")
    playback = simpleaudio.play_buffer(
        audio,
        num_channels=1,
        sample_rate=hp.sample_rate
    )
##
# CONFIG
##

DEVICE_ID = 6
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

            rec = vosk.KaldiRecognizer(model, args.samplerate)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    r = rec.Result()
                    print("result: {}".format(r['text']))
                    answer = generate_text(r['text'])
                    audio = generate_audio(answer)
                else:
                    pass
                    #print(rec.PartialResult())

except KeyboardInterrupt:
    print('\nDone')
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
    exit(-1)
