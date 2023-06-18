import librosa, soundfile
import matplotlib.pyplot as plt
import numpy as np
from time import time
from collections import namedtuple
import speedup

# data -- [-1, 1] array
# sample_rate -- num samples per second, integer
Sound = namedtuple("Sound", ["data", "sample_rate"])

file_name = "violin"
#file_name = "let_it_be"
#file_name = "yellow_submarine"
#file_name = "my_favorite_things"
in_file_path = f"samples/original/{file_name}.wav"
out_file_path = f"samples/{file_name}" + "_{0}.wav"

sound, sample_rate = Sound(*librosa.load(in_file_path))

# sound -- single channel (mono) numpy 1D array of PCM data (like in wav format)
# sample_rate -- samples per second

for algorithm in [speedup.speedup_pitch_aware, speedup.algorithm2]:
    SS = time()
    out = algorithm(sound, sample_rate)
    total_time = time() - SS
    print(f"algorithm={algorithm.__name__} {total_time=}")
    soundfile.write(out_file_path.format(algorithm.__name__), out, sample_rate)
