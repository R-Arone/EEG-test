# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:08:29 2018

@author: Rafael Arone
"""
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np
#df2 = pd.read_csv('openBCI_raw_2014-10-04_19-06-13_O1_Alpha.txt', sep=',', header = None)
df2 = pd.read_csv('OpenBCI-RAW-teste_Rafael.txt', sep=',', header = None)
df2.drop(df2.columns[[0,9,10,11,12]], axis=1, inplace=True)
#df2.drop(df2.columns[0], axis=1, inplace=True)

ch = 5
valor = df2[ch].values
valor = valor[1:-1]
valor = sg.detrend(valor)
plt.plot(valor)
plt.title('Sinal Temporal')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [uV]')
plt.figure()
SampleRate = 250
f, Pxx_den = sg.welch(valor,SampleRate)
plt.semilogy(f, Pxx_den)
plt.title('Welch Periodogram')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

f1, f2 = 5, 55
w = sg.firwin(256, [f1, f2], pass_zero=False, fs = 250,window="hann")
con = sg.convolve(w,valor)
plt.figure()
f, Pxx_den = sg.welch(con,SampleRate)
plt.plot(f, Pxx_den)
plt.xlim(5,55)
plt.title('Welch Periodogram with filter')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.figure()

SamplesPerInteraction = 512
window = sg.windows.dpss(SamplesPerInteraction, 2.5)
f, t, Zxx = sg.stft(valor,
                                           nperseg=SamplesPerInteraction, noverlap=None,
                                           fs=SampleRate, window=window, boundary=None )
plt.pcolormesh(t, f, np.abs(Zxx))
plt.title('Colormash - STFT')
plt.figure()
plt.specgram(con,  Fs=SampleRate,cmap=plt.cm.gist_heat)

plt.title('Spectogram')
plt.show()
#Fazendo a filtragem do sinal:

valores = df2.values[1:-1]
filtred_signal_transpose = np.transpose(valores)
def filter_signal(filtered):
    w = sg.firwin(256, [4, 50], pass_zero=False, fs = 250, window="hann")
    w=np.flip(w) # Passar o mais w0 para a ultima posicao para multiplicar o ultimo dado que chegou da fila
    p = []
    for j in filtered:
        p.append(sg.convolve(j,w))
    return p

filtred_signal_transpose = filter_signal(filtred_signal_transpose)

def PowerSpectre(FTMatrix, step, fcutoff1, fcutoff2):
    # This function creates a value of power spectre using the Short-Time Fourier Transform
    # To get the matriz FTMatriz t x f, and calculate the step, which represents the step
    Window = FTMatrix[int(np.floor(fcutoff1 / step)):int(np.floor(fcutoff2 / step)), :]
    Power = np.sum(np.abs(Window[:]) ** 2, axis=0)
    return Power / (len(Power))  # The division for len(Power) server for normalize the power spectre
numberofchannels = 8
#filtred_signal_transpose = np.transpose(df2.values)
def PowerSpec(filtred_signal_transpose,numberofchannels):
    Xpowerspec = []
    cutoff = [1.0, 4.0, 8.0, 16.0, 32.0]
    for channels in range(numberofchannels):
            value = [value / 10 ** 6 for value in filtred_signal_transpose[channels]]  # Change the scale of the data
            frequency, time, Zxx = sg.stft(value,
                                           nperseg=SamplesPerInteraction, noverlap=None,
                                           fs=SampleRate, window=window, boundary=None )
            # plotSTFTdata(frequency, time, Zxx, channels)
            # Using the cutoff frequencies defined before in the script
            f_step = frequency[-1] / (len(frequency) - 1)
            DensityPower = [PowerSpectre(Zxx, f_step, cutoff[0], cutoff[-1])]
            for bands in range(len(cutoff) - 1):
                Density = [PowerSpectre(Zxx, f_step, cutoff[bands], cutoff[bands + 1])]
                DensityPower = np.concatenate((DensityPower, Density), axis=0)
            Xpowerspec.append([DensityPower])
    return Xpowerspec

Xpowerspec = PowerSpec(filtred_signal_transpose,numberofchannels)
#Calculando a razão entre a potência máxima da banda e de cada uma das bandas do eeg
def Norm(Xpowerspec):
    Xpowerspec_norm = []
    for X in Xpowerspec:
        # print(len(X[0][0]))
        Total = X[0][0]
        Xnorm = []
        for i in range(len(X[0]) - 1):
            Xnorm.append([X[0][i + 1][j] / Total[j] for j in range(len(Total))])
        Xpowerspec_norm.append(Xnorm)
    return Xpowerspec_norm

Xpowerspec_norm = Norm(Xpowerspec)
for i in range(8):
#Plot each channel at matplotlib
    plt.figure()
    [plt.plot(k) for k in Xpowerspec_norm[i]]
    plt.legend(['Delta','Theta','Alpha','Beta'])
    plt.title('Porcentagem de cada banda no canal '+str(i+1))
    plt.xlabel('Tempo')
    plt.ylabel('Porcentagem da potência[%]')
plt.show()