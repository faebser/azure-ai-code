#!/usr/bin/env python3

import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import datetime
import json
import traceback
from collections import deque
import editdistance
from time import sleep

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

def check_for_prerecored_answer(_question):
    _q = _question.lower()
    return ( editdistance.eval(_q, QUESTION) <= 2 or editdistance.eval(_q, QUESTION2) <= 2)

def get_prerecorded_answer(_question):
    _q = _question.lower()
    qq = [ (QUESTION, 'Je suis Lissa'), (QUESTION2, 'partout, prend un mirroir.') ]
    for _qq, _a in qq:
        if editdistance.eval(_q, _qq): return _a

def clean_text(_text):
    return _text.replace("\n", " ").strip()

def generate_text(question, name, _context):
    prompt = "{}A: {}\n B: ".format(tokenizer.bos_token, question)
    index = 0

    original_prompt = prompt

    while len(prompt) < 500 and len(_context) -1 > index:
        prompt = _context[ index ] + ' ' + prompt
        index = index + 1

    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    to_return = []
    real_r = ''
    tries = 1

    # it will try MAX_TRIES times
    while len(real_r) < 1 and tries <= MAX_TRIES:

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
            b = o.split( 'B:' )[-1]
            to_return.append(b)

            # sort by length and return only the longest
            r = sorted(to_return, key=len)

        real_r = clean_text(r[-1])

    # check if it returned an empty string and max tries
    if len(real_r) == 0 and tries == MAX_TRIES:
        real_r = "Pardon, je ne sait pas"
    else:
        _context.appendleft("A: {} B: {}\n".format(question, real_r))

    return real_r, _context

def generate_audio(answer):
    sentences = [ answer ]
    print(sentences)
    spectograms = [ synthesize(model_taco, "|" + s + ACCENT) for s in sentences if len(s) > 0 ]
    return [ audio.inverse_spectrogram(_s, not hp.predict_linear) for _s in spectograms ]

def play_audio(speech_audios, name):
    _name = os.path.join('recordings', name)
    files = []
    for index, segment in enumerate(speech_audios):
        audio.save(segment, _name + str(index) + '.wav')
        files.append( _name + str(index) + '.wav')
    #_name = os.path.join('recordings', name) +'.mp3'

    for _n in files:
        talk = AudioSegment.from_file(_n, format='wav')
        play(talk)
    #tts = gTTS(answer, lang='fr')
    #tts = gTTS(answer, lang='fr', slow=True)
    #tts.save(_name)
    #talk = AudioSegment.from_file(_name, format='mp3')

def save(question, answer, name):
    with open(os.path.join('recordings', name) + '.txt', 'w') as output_file:
        output_file.write(question)
        output_file.write('\n\n')
        output_file.write(answer)
##
# CONFIG
##

DEVICE_ID =  "default"
# get device from vosk example: python test_microphone.py -l
BLOCK_SIZE = 80000
SAMPLE_RATE = None # set to None for auto samplerate
ACCENT = "|00-de|fr"
QUESTION = "Qui est tu".lower()
QUESTION2 = "ou es tu".lower()
MAX_TRIES = 2

if SAMPLE_RATE is None:
    device_info = sd.query_devices(DEVICE_ID, 'input')
    # soundfile expects an int, sounddevice provides a float:
    SAMPLE_RATE = int(device_info['default_samplerate'])

q = queue.Queue()
context = deque(maxlen=10)

try:
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize = BLOCK_SIZE, device=DEVICE_ID,
                           dtype='int16', channels=1, callback=callback):

            rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    #r = rec.Result()
                    r = json.loads(rec.Result())
                    print("result with len {}: {}".format(len(r['text']), r['text']))
                    if len(r['text']) > 3:
                        name = date_name()
                        if check_for_prerecored_answer(r['text']):
                            # we found a prerecorded questions
                            answer = get_prerecorded_answer(r['text'])
                            sleep(5)
                        else:
                            # no prerecored answer
                            answer, context = generate_text(r['text'], name, context)
                            print("new context:")
                            print(context)
                            answer = clean_text(answer)
                        _audios = generate_audio(answer)
                        play_audio(_audios, name)
                        save(r['text'], answer, name)
                        with q.mutex: q = queue.Queue()
                    else:
                        print("found result but length was shorter than 3, so no generation")
                else:
                    pass
                    #print(rec.PartialResult())

except KeyboardInterrupt:
    print('\nDone')
except Exception as e:
    traceback.print_tb(e.__traceback__)
    print(type(e).__name__ + ': ' + str(e))
    exit(-1)
