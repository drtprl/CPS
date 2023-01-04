import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

angulo_S1 = 90
path_sons = './sons/'
path_HRTF = './HRTF/elev0/'
filenameSom = 'scissor-snips.wav'
Fs, xSom = wav.read(path_sons + filenameSom)
xSom = xSom[0:100000]
print(path_sons + filenameSom)
print(xSom)
print(Fs)

#H0e035a.wav
HRTFname = 'H0e'+ '0' + str(angulo_S1) + 'a.wav'
Fs, xHRTF = wav.read(path_HRTF + HRTFname)
print(path_sons + filenameSom)
print(xHRTF)
print(Fs)

# Converter os sinais para valores de 32 bits,
# para garantir que não existe overflow quando se faz a convolução
xSom = np.asarray(xSom, dtype=np.int32)
xHRTF = np.asarray(xHRTF, dtype=np.int32)

yL = np.convolve(xSom, xHRTF[:, 0])
yR = np.convolve(xSom, xHRTF[:, 1])

y = np.empty((2, len(yL)), dtype=np.int32)

y[0] = yL/(2**15) #Tem que se dividir por metade da gama dinâmica para depois se converter para 16 bits (int16)
y[1] = yR/(2**15)

y = np.transpose(y)

print(y)
plt.figure()
plt.plot(y)
plt.show()

#===== Reverberação
RT60 = 0.1
n = np.arange(0, 1*Fs)
C = 3*np.log(10/RT60)

beta = np.random.randn(len(n))
xRIR = beta * np.exp(-C/Fs*n) # Room Impulse Response

xRIR = xRIR/np.max(np.abs(xRIR)) # normaliza a 1
xRIR = xRIR*2**(15-1) # normaliza a int16
xRIR = np.asarray(xRIR, dtype=np.int32)

yL_Rev = np.convolve(xRIR, yL)
yR_Rev = np.convolve(xRIR, yR)
print(yL_Rev)
y_Rev = np.empty((2, len(yL_Rev)), dtype=np.int32)

y_Rev[0] = yL_Rev/(2**15) #Tem que se dividir por metade da gama dinâmica para depois se converter para 16 bits (int16)
y_Rev[1] = yR_Rev/(2**15)

y_Rev = np.transpose(y_Rev)
print(y_Rev)

plt.figure()
plt.plot(xRIR)
plt.show()

wav.write('TestBinauralHRTF_90.wav', Fs, y.astype(np.int16))

wav.write('TestBinauralHRTF_90_Rev.wav', Fs, y_Rev.astype(np.int16))
