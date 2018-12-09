import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np
import csv
import io
#with open('openBCI_raw_2014-10-04_18-55-41_O1_Alpha.txt') as f:
#openBCI_raw_2014-10-04_19-06-13_O1_Alpha.txt
#data_2013-11-04_20-31-45_MuWaves2_HomebrewElectrodes_arbitraryScale100000.csv
df2 = pd.read_csv('openBCI_raw_2014-10-04_19-06-13_O1_Alpha.txt', sep=',', header = None)
df2.drop(df2.columns[0], axis=1, inplace=True)
df = pd.read_csv('openBCI_raw_2014-10-05_17-14-45_O1_Alpha_noCaffeine.txt', sep=',', header = None)
df.drop(df.columns[0], axis=1, inplace=True)
#df2 = pd.read_csv('data_2013-11-04_20-31-45_MuWaves2_HomebrewElectrodes_arbitraryScale100000.csv', sep=',', header = None)
#df2 = pd.read_csv('Dados_sem_gel.txt', sep = '   ',header = None)
ch = 2
valor = df2[ch].values
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

f1, f2 = 1, 55
w = sg.firwin(256, [f1, f2], pass_zero=False, fs = 250,window="hann")
con = sg.convolve(w,valor)
plt.figure()
f, Pxx_den = sg.welch(con,SampleRate)
plt.semilogy(f, Pxx_den)
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

filtred_signal_transpose = np.transpose(df2.values)
filtred_signal_transpose2 = np.transpose(df.values)
def filter_signal(filtered):
    w = sg.firwin(256, [1, 50], pass_zero=False, fs = 250, window="hann")
    w=np.flip(w) # Passar o mais w0 para a ultima posicao para multiplicar o ultimo dado que chegou da fila
    p = []
    for j in filtered:
        p.append(sg.convolve(j,w))
    return p

filtred_signal_transpose = filter_signal(filtred_signal_transpose)
filtred_signal_transpose2 = filter_signal(filtred_signal_transpose2)

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
Xpowerspec2 = PowerSpec(filtred_signal_transpose2,numberofchannels)
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
Xpowerspec_norm2 = Norm(Xpowerspec2)

#Plot each channel at matplotlib
plt.figure()
[plt.plot(k) for k in Xpowerspec_norm[ch]]
plt.legend(['Delta','Theta','Alpha','Beta'])
plt.title('Relação da potência da banda pela total')
plt.xlabel('Tempo')
plt.ylabel('razão')
plt.show()

#Plot Theta/Alpha
plt.figure()
new = [Xpowerspec_norm[ch][1][k]/Xpowerspec_norm[ch][2][k] for k in range(len(Xpowerspec_norm[ch][0]))]
plt.plot(new)
plt.title('Theta/Alpha')
plt.xlabel('Tempo')
plt.ylabel('Porcentagem da potência[%]')
plt.show()


#Training Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
def train_matrix(Xpowerspec):
    k = []
    for i in Xpowerspec:
        for j in i:
            k.append(j)
    return(k)

matrix_train = train_matrix(Xpowerspec_norm)
matrix_train2 = train_matrix(Xpowerspec_norm2)
#print(matrix_train)
#print(len(matrix_train))
hiddenlayer = 35
#hiddenlayer2 = 7
def train(hiddenlayer):
    MultiLayerPerceptron = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(hiddenlayer,),
                                                  random_state=1, max_iter=10000,activation = 'logistic')

    #Realizando o treinamento
    #Como os valores já estão normalizados (0 - 1), não há a necessidade de fazer a normalização
    y1 = np.zeros(len(matrix_train[0]))
    #y[390:480] = 1 #Teste para o treinamento
    y2 = np.ones(len(matrix_train2[0]))
    matrix_train = np.concatenate((matrix_train,matrix_train2),axis = 1)
    y = np.concatenate((y1,y2),axis = 0)
    #Separando em dados de teste e treinamento
    X_train, X_test, y_train, y_test = train_test_split(np.transpose(matrix_train), y, test_size=.5)

    MultiLayerPerceptron.fit(X_train, y_train)
    #Parâmetros após o teste:
    print(MultiLayerPerceptron.coefs_)
    #Teste:
    y_pred = MultiLayerPerceptron.predict(X_test)

    #Confusion Matrix, para verificar a taxa de acerto:
    score =  MultiLayerPerceptron.score(X_test,y_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(score)
    print(cm)

train(hiddenlayer)