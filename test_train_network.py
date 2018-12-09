import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import io

def PowerSpectre(FTMatrix, step, fcutoff1, fcutoff2):
    # This function creates a value of power spectre using the Short-Time Fourier Transform
    # To get the matriz FTMatriz t x f, and calculate the step, which represents the step
    Window = FTMatrix[int(np.floor(fcutoff1 / step)):int(np.floor(fcutoff2 / step)), :]
    Power = np.sum(np.abs(Window[:]) ** 2, axis=0)
    return Power / (len(Power))  # The division for len(Power) server for normalize the power spectre

#with open('openBCI_raw_2014-10-04_18-55-41_O1_Alpha.txt') as f:
#openBCI_raw_2014-10-04_19-06-13_O1_Alpha.txt
#data_2013-11-04_20-31-45_MuWaves2_HomebrewElectrodes_arbitraryScale100000.csv
#openBCI_raw_2014-10-05_17-14-45_O1_Alpha_noCaffeine.txt
#df2 = pd.read_csv('data_2013-11-04_20-31-45_MuWaves2_HomebrewElectrodes_arbitraryScale100000.csv', sep=',', header = None)
SampleRate = 250
SamplesPerInteraction = 512
numberofchannels = 8
window = sg.windows.dpss(SamplesPerInteraction, 2.5)
df2 = pd.read_csv('Dados_sem_gel.txt', sep = '   ',header = None)
time = df2[8].values
df2 = df2.drop(8,1)
valor = []
[valor.append(df2[a].values) for a in range(8)]
valor = sg.detrend(valor)
[plt.plot(time, a) for a in valor]
#[plt.plot(a) for a in valor]
plt.show()
cutoff = [1.0, 4.0, 8.0, 16.0, 32.0]
filteredsignal = valor
#agora:
Xpowerspec = []
# Time - Frequency Transform
# In this section, fot transforming time-frequency, we'll utilise the short-time fourier transform
for channels in range(numberofchannels):
    value = [valor[channels] / 10 ** 6 for value in filteredsignal]  # Change the scale of the data
    frequency, time, Zxx = sg.stft(value,
                                   nperseg=SamplesPerInteraction, noverlap=None,
                                   fs=SampleRate, window=window, boundary=None)
    # self.plotSTFTdata(frequency, time, Zxx, channels)
    # Using the cutoff frequencies defined before in the script
    f_step = frequency[-1] / (len(frequency) - 1)
    DensityPower = [PowerSpectre(Zxx, f_step, cutoff[0], cutoff[-1])]
    for bands in range(len(cutoff) - 1):
        Density = [PowerSpectre(Zxx, f_step, cutoff[bands], cutoff[bands + 1])]
        DensityPower = np.concatenate((DensityPower, Density), axis=0)
    Xpowerspec.append([DensityPower])
#print(Xpowerspec)
plt.figure()
print(len(Xpowerspec))
print(len(Xpowerspec[0]))
print(len(Xpowerspec[0][0]))
print(len(Xpowerspec[0][0][0]))
[plt.plot(a[0][0][0]) for a in Xpowerspec]
plt.show()


#Trainning neural network:

