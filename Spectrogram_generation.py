# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Read the input file
with open('D:\Docs\B.Tech_Project_Final_year\Project submit\Biospectral_classification\Sequences\Train\seq511.dat', 'r') as file:
    x = file.readlines()

# Convert the list of strings to a single string
a = ''.join(x).replace('\n', '')

# Initialize the combined string
c = []

# Append characters to c
for i1 in range(len(a)):
    c.append(a[i1])

# Generate numeric code for text-data
D_length = len(c)
I6 = np.zeros(D_length-1)

for i in range(D_length-1):
    code = c[i+1]
    if code in {'A', 'a'}:
        I6[i] = 1
    elif code in {'G', 'g'}:
        I6[i] = 2
    elif code in {'C', 'c'}:
        I6[i] = 3
    elif code in {'T', 't'}:
        I6[i] = 4


# Compute and plot the spectrogram with adjusted nperseg and noverlap
nperseg = 50
noverlap = 25
f, t, Sxx = spectrogram(I6, fs=1.0, nperseg=nperseg, noverlap=noverlap)

plt.pcolormesh(t, f, np.log(Sxx + 1e-10))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar(label='Log intensity')
plt.show()