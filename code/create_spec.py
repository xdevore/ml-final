import os
import matplotlib.pyplot as plt
import numpy as np

#for loading and visualizing audio files
import librosa
import librosa.display

from pydub import AudioSegment
from tempfile import mktemp
import soundfile as sf
import math
import random


os.chdir("/homes/xdevore/ml-final-project/ml-final/code/house_songs_test")
os.system("spotdl https://open.spotify.com/playlist/45Ydky39Ai4C5ZE3fF0Kh3?si=6aad261538db4af7")

# audio_fpath = "/homes/xdevore/ml-final-project/ml-final/code/rock_songs_test"
# audio_clips = os.listdir(audio_fpath)
# print("No. of .mp3 files in audio folder = ",len(audio_clips))
#
# audio_fpath = "/homes/xdevore/ml-final-project/ml-final/code/rap_songs_test"
# audio_clips = os.listdir(audio_fpath)
# print("No. of .mp3 files in audio folder = ",len(audio_clips))
#
# audio_fpath = "/homes/xdevore/ml-final-project/ml-final/code/jazz_songs_test"
# audio_clips = os.listdir(audio_fpath)
# print("No. of .mp3 files in audio folder = ",len(audio_clips))
#
# audio_fpath = "/homes/xdevore/ml-final-project/ml-final/code/house_songs_test"
# audio_clips = os.listdir(audio_fpath)
# print("No. of .mp3 files in audio folder = ",len(audio_clips))


# start = 43
#
# for i in range(start, len(audio_clips)):
#
#     print(i)
#
#
# sound = AudioSegment.from_mp3(audio_fpath)
# wav_file = mktemp('.wav')
# sound.export(wav_file, format="wav")
#
# x, data = librosa.load(wav_file, sr=22050)
#
# duration = librosa.get_duration(y=x)
# duration = duration - duration % 30
#
# discard_samples = int(len(x) - duration * 22050)
# discard_samples_end = 0
# if discard_samples % 2 == 0:
#     discard_samples = discard_samples/2
#     discard_samples_end = len(x) - discard_samples
# else:
#     discard_samples = math.floor(discard_samples/2)
#     discard_samples_end = len(x) - discard_samples - 1
#
# x = x[int(discard_samples):int(discard_samples_end)]
#
# num_specs = duration / 30
#
# segments = np.split(x, num_specs)
#
# specs_array = []
#
# if num_specs < 4:
#     for i in range(num_specs):
#         specs_array.append(i)
# else:
#     for i in range(3):
#         segment_index = random.randint(0, num_specs - 1)
#         if segment_index not in specs_array:
#             specs_array.append(segment_index)
#         else:
#             while segment_index not in specs_array:
#                 segment_index = random.randint(0, num_specs - 1)
#             specs_array.append(segment_index)
#
# print(specs_array)
#
# for i, segment in enumerate(segments):
#
#     if i in specs_array:
#         print(i)
#
#         X = librosa.stft(segment)
#         Xdb = librosa.amplitude_to_db(abs(X))
#
#         plt.figure(figsize=(14, 5))
#
#         librosa.display.specshow(Xdb, x_axis='time', y_axis='log')
#
#         plt.axis("off")
#         plt.savefig("log" + str(i) + ".jpg")
#         plt.close()
#         X = None
#         Xdb = None
