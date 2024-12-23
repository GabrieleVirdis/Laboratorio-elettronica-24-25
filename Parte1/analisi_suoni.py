# Import Moduli
import numpy as np
import pandas as pd
from scipy import constants, fft
import matplotlib.pyplot as plt

# Lettura dati
df1 = pd.read_csv(
    'https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/data1.txt',
    sep="\t", header=None
)

df2 = pd.read_csv(
    'https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/data2.txt',
    sep="\t", header=None
)

df3 = pd.read_csv(
    'https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/data3.txt',
    sep="\t", header=None
)

# Differenza temporale
time_step = np.mean(np.diff(df1.iloc[:, 0]))  # primo dataset 
time_step2 = np.mean(np.diff(df2.iloc[:, 0])) # secondo dataset
time_step3 = np.mean(np.diff(df3.iloc[:, 0])) # terzo dataset 

# Segnali
signal = df1.iloc[:, 1].values # primo dataset
signal2 = df2.iloc[:, 1].values # secondo dataset
signal3 = df3.iloc[:, 1].values # terzo dataset

# Funzione per trasformata di Fourier discreta
# Trasformata di Fourier
fft_coeffs = 0.5 * fft.fft(signal)  # primo dataset 
freqs = 0.5 *fft.fftfreq(len(signal), d=time_step)  # primo dataset

fft_coeffs2 = 0.5 * fft.fft(signal2)  # secondo dataset 
freqs2 = 0.5 *fft.fftfreq(len(signal2), d=time_step)  # secondo dataset

fft_coeffs3 = 0.5 * fft.fft(signal3)  # terzo dataset 
freqs3 = 0.5 *fft.fftfreq(len(signal3), d=time_step)  # terzo dataset

# Filtraggio: Maschera
fft_filtered = fft_coeffs.copy() #df1
mask = (freqs >= 200) & (freqs <= 240) #df1
fft_filtered[~mask] = 0 #df1

fft_filtered2 = fft_coeffs2.copy() #df2
mask2 = (freqs >= 495) & (freqs <= 505) #df2
fft_filtered2[mask2] = 0 #df2

fft_filtered3 = fft_coeffs3.copy() #df3
mask3 = (freqs >= 0) & (freqs <= 500) #df3
fft_filtered3[mask3] = 0 #df3

# Antitrasformata di fourier 
signal_reconstructed = fft.ifft(fft_coeffs, n=len(signal)) #antit. su coefficenti originali df1
signal_filtered = fft.ifft(fft_filtered, n=len(signal)) #antit. su coefficenti filtrati df1
 
signal_reconstructed2 = fft.ifft(fft_coeffs2, n=len(signal2)) #antit. su coefficenti originali df2
signal_filtered2 = fft.ifft(fft_filtered2, n=len(signal2)) #antit. su coefficenti filtrati df2
 
signal_reconstructed3 = fft.ifft(fft_coeffs3, n=len(signal3)) #antit. su coefficenti originali df3
signal_filtered3 = fft.ifft(fft_filtered3, n=len(signal3)) #antit. su coefficenti filtrati df3

# Plot
fig, axs = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)

# Plot del segnale filtrato e ricostruito
axs[0, 0].plot(df1.iloc[:, 0], signal_reconstructed, alpha=0.7, color='green', label='Segnale originale')
axs[0, 0].plot(df1.iloc[:, 0], signal_filtered, color='purple', label='Segnale filtrato')
axs[0, 0].set_title('Sintesi del segnale [data1]')
axs[0, 0].set_xlabel('Tempo')
axs[0, 0].set_ylabel('Ampiezza')
axs[0, 0].legend()
axs[0, 0].set_xlim(0.57 , 0.59)

# Plot della FFT (reale)
axs[1, 0].plot(freqs[:int(freqs.size/2)], np.real(np.abs(fft_coeffs[:int(fft_coeffs.size/2)])), color='black', label=r'$|X_k| [parte reale]$')
axs[1, 0].set_title('Coefficenti di Fourier reali [data1]')
axs[1, 0].set_xlabel('Frequenza [Hz]')
axs[1, 0].set_ylabel(r'$|X_k|$')
axs[1, 0].legend()

# Plot della FFT (immaginaria)
axs[2, 0].plot(freqs[:int(freqs.size/2)], np.imag(fft_coeffs[:int(fft_coeffs.size/2)]), color='orange', label=r'$X_k$ [parte immaginaria]')
axs[2, 0].set_title('Coefficenti di Fourier immaginari [data1]')
axs[2, 0].set_xlabel('Frequenza [Hz]')
axs[2, 0].set_ylabel(r'$X_k$')
axs[2, 0].legend()

# Plot del segnale filtrato e ricostruito dataset 2
axs[0, 1].plot(df2.iloc[:, 0], signal_reconstructed2, alpha=0.7, color='green', label='Segnale originale')
axs[0, 1].plot(df2.iloc[:, 0], signal_filtered2, color='purple', label='Segnale filtrato')
axs[0, 1].set_title('Sintesi del segnale [data2]')
axs[0, 1].set_xlabel('Tempo')
axs[0, 1].set_ylabel('Ampiezza')
axs[0, 1].legend()
axs[0, 1].set_xlim(0.54 , 0.68)

# Plot della FFT (reale) dataset 2
axs[1, 1].plot(freqs2[:int(freqs2.size/2)], np.real(np.abs(fft_coeffs2[:int(fft_coeffs2.size/2)])), color='black', label=r'$|X_k|$ [parte reale]')
axs[1, 1].set_title('Coefficenti di Fourier reali [data2]')
axs[1, 1].set_xlabel('Frequenza [Hz]')
axs[1, 1].set_ylabel(r'$|X_k|$')
axs[1, 1].legend()

# Plot della FFT (immaginaria) dataset 2
axs[2, 1].plot(freqs2[:int(freqs2.size/2)], np.imag(fft_coeffs2[:int(fft_coeffs2.size/2)]), color='orange', label=r'$X_k$ [parte immaginaria]')
axs[2, 1].set_title('Coefficenti di Fourier immaginari [data2]')
axs[2, 1].set_xlabel('Frequenza [Hz]')
axs[2, 1].set_ylabel(r'$X_k$')
axs[2, 1].legend()

# Plot del segnale filtrato e ricostruito dataset 3
axs[0, 2].plot(df3.iloc[:, 0], signal_reconstructed3, alpha=0.7, color='green', label='Segnale originale')
axs[0, 2].plot(df3.iloc[:, 0], signal_filtered3, color='purple', label='Segnale filtrato') 
axs[0, 2].set_title('Sintesi del segnale [data3]')
axs[0, 2].set_xlabel('Tempo')
axs[0, 2].set_ylabel('Ampiezza')
axs[0, 2].legend()
axs[0, 2].set_xlim(0.577 , 0.579)

# Plot della FFT (reale) dataset 3
axs[1, 2].plot(freqs3[:int(freqs3.size/2)], np.real(np.abs(fft_coeffs3[:int(fft_coeffs3.size/2)])), color='black', label=r'$X_k$ [parte reale]')
axs[1, 2].set_title('Coefficenti di Fourier reali (data3)')
axs[1, 2].set_xlabel('Frequenza [Hz]')
axs[1, 2].set_ylabel(r'$|X_k|$')
axs[1, 2].legend()

# Plot della FFT (immaginaria) dataset 3
axs[2, 2].plot(freqs3[:int(freqs3.size/2)], np.imag(fft_coeffs3[:int(fft_coeffs3.size/2)]), color='orange', label=r'$X_k$ [parte immaginaria]')
axs[2, 2].set_title('Coefficenti di Fourier immaginari (data3)')
axs[2, 2].set_xlabel('Frequenza [Hz]')
axs[2, 2].set_ylabel(r'$X_k$')
axs[2, 2].legend()

plt.show()

# Plot zoommato del filtro del dataset3

plt.plot(df3.iloc[:, 0], signal_reconstructed3, alpha=0.7, color='green', label='Segnale originale')
plt.plot(df3.iloc[:, 0], signal_filtered3, color='purple', label='Segnale filtrato')
plt.xlabel('Tempo')
plt.ylabel('Ampiezza')
plt.xlim(0.57758 , 0.57768)
plt.ylim(0.3 , 0.45)
plt.show()

# Plot della potenza spettrale df1
fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True) 

axs[0,0].plot(freqs[:int(freqs.size/2)], np.absolute(fft_coeffs[:int(fft_coeffs.size/2)])**2, color='red')
axs[0,0].set_title('Potenza Spettrale [data1]')
axs[0,0].set_xlabel('Frequenza [Hz]')
axs[0,0].set_ylabel(r'$|X_k|^2$')

#zoom1
ins_ax = axs[0,0].inset_axes([0.75, 0.75, 0.2, 0.2])  # [x, y, width, height]
ins_ax.plot(freqs[:int(freqs.size/2)], np.absolute(fft_coeffs[:int(fft_coeffs.size/2)])**2, color='red')
ins_ax.set_ylim(0, 6*10**8)
ins_ax.set_xlim(0 , 50)

axs[0, 1].plot(freqs2[:int(freqs2.size/2)], np.absolute(fft_coeffs2[:int(fft_coeffs2.size/2)])**2, color='blue')
axs[0,1].set_title('Potenza Spettrale [data2]')
axs[0,1].set_xlabel('Frequenza [Hz]')
axs[0,1].set_ylabel(r'$|X_k|^2$')

#zoom2
ins_ax = axs[0, 2].inset_axes([0.75, 0.75, 0.2, 0.2])  # [x, y, width, height]
ins_ax.plot(freqs2[:int(freqs2.size/2)], np.absolute(fft_coeffs2[:int(fft_coeffs2.size/2)])**2, color='orange')
ins_ax.set_ylim(0, 1.5*10**8)
ins_ax.set_xlim(0, 10)

axs[0, 2].plot(freqs3[:int(freqs3.size/2)], np.absolute(fft_coeffs3[:int(fft_coeffs3.size/2)])**2, color='orange')
axs[0, 2].set_title('Potenza Spettrale [data3]')
axs[0, 2].set_xlabel('Frequenza [Hz]')
axs[0, 2].set_ylabel(r'$|X_k|^2$')

#zoom3
ins_ax = axs[0,2].inset_axes([0.75, 0.75, 0.2, 0.2])  # [x, y, width, height]
ins_ax.plot(freqs3[:int(freqs3.size/2)], np.absolute(fft_coeffs3[:int(fft_coeffs3.size/2)])**2, color='orange')
ins_ax.set_ylim(0, 1.5*10**8)
ins_ax.set_xlim(0 , 10)

# Plot della potenza spettrale filtrata

axs[1, 0].plot(freqs[:int(freqs.size/2)], np.abs(fft_filtered[:int(fft_filtered.size/2)])**2, color='red', label=r'Filtro $f \in [200,240] Hz$') # df1 filtrata
axs[1, 0].set_title('Potenza Spettrale filtrata [data1]')
axs[1, 0].set_xlabel('Frequenza [Hz]')
axs[1, 0].set_ylabel(r'$|X_k|^2$')
axs[1, 0].legend()

axs[1, 1].plot(freqs2[:int(freqs2.size/2)], np.abs(fft_filtered2[:int(fft_filtered2.size/2)])**2, color='blue', label=r'Filtro: $f \in [495,505] Hz$') # df2 filtrata
axs[1, 1].set_title('Potenza Spettrale filtrata [data2]')
axs[1, 1].set_xlabel('Frequenza [Hz]')
axs[1, 1].set_ylabel(r'$|X_k|^2$')
axs[1, 1].legend()

axs[1, 2].plot(freqs3[:int(freqs3.size/2)], np.abs(fft_filtered3[:int(fft_filtered3.size/2)])**2, color='orange', label=r'Filtro: $f \leqslant 500 Hz$') # df3 filtrata
axs[1, 2].set_title('Potenza Spettrale filtrata [data3]')
axs[1, 2].set_xlabel('Frequenza [Hz]')
axs[1, 2].set_ylabel(r'$|X_k|^2$')
axs[1, 2].legend()

plt.show()


