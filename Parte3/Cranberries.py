# Import Moduli
import numpy as np
from scipy import fft, constants
import matplotlib.pyplot as plt
import soundfile as sf

# Leggi il file audio (diapason.wav)
data, samplerate = sf.read('/home/gabriele/Downloads/primo.wav')
print('-------------------------------')
print('Samplerate:', samplerate)
print('Array:', data)
print('Dimensioni Array:', len(data))
print('-------------------------------')

# Creazione nuovo file array (new_diapason.wav)
sf.write('/home/gabriele/Downloads/new_primo.wav', data, samplerate)
new_data, new_samplerate = sf.read('/home/gabriele/Downloads/new_primo.wav')

# solo un canale
if len(new_data.shape) > 1:
    new_data = new_data[:, 0] 


# Trasformata di Fourier
fft_coeffs = fft.fft(new_data)
diff_temp = 1 / new_samplerate
freqs =  fft.fftfreq(len(new_data), diff_temp)


# Filtraggi: Maschere 
fft_filtered = fft_coeffs.copy() 
mask = (freqs >= 0) & (freqs <= 500) # Filtro per il basso 
fft_filtered[~mask] = 0 

fft_filtered1 = fft_coeffs.copy() 
mask1 = (freqs >= 500) & (freqs <= 5000) # Filtro per la chitarra 
fft_filtered1[~mask1] = 0 

# Antitrasformata di fourier per segnale filtrato
anti_fft_o = fft.ifft(fft_coeffs)
anti_fft = fft.ifft(fft_filtered)
anti_fft1 = fft.ifft(fft_filtered1)

# Plot dati
fig, axs = plt.subplots(1, 3, figsize=(10, 6), layout='constrained')

# Segnale originale
axs[0].plot(new_data, color='green')
axs[0].set_xlabel('Tempo [s]')
axs[0].set_ylabel('Ampiezza')
axs[0].set_title(r'Segnale originale [$\it{The \ Cranberries\ -\ Zombie}$]')
axs[0].legend(['Segnale originale'], fontsize=10)

# Coefficienti di Fourier (parte reale)
axs[1].plot(freqs[:len(freqs)//2], ((fft_coeffs[:len(fft_coeffs)//2])).real, color='yellow')
axs[1].set_title('Parte reale dei coefficienti di Fourier')
axs[1].set_xlabel('Frequenza [Hz]')
axs[1].set_ylabel(r'$|X_k|$')
axs[1].legend(['X_k [parte reale]'], fontsize=10)

# Coefficienti di Fourier (parte immaginaria)
axs[2].plot(freqs[:len(freqs)//2], np.imag(fft_coeffs[:len(fft_coeffs)//2]), color='orange')
axs[2].set_title('Parte immaginaria dei coefficienti di Fourier')
axs[2].set_xlabel('Frequenza [Hz]')
axs[2].set_ylabel(r'$X_k$')
axs[2].legend(['X_k [parte immaginaria]'], fontsize=10)


# Potenza spettrale e confronto segnali filtrati e originale
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_coeffs[:len(fft_coeffs)//2])**2, color='green') #originale
axs[0].set_title(r'Potenza spettrale originale [$\it{The \ Cranberries\ -\ Zombie}$]')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')

axs[1].plot(anti_fft_o, color='green', label='Segnale originale') #ricostruzione segnale originale
axs[1].set_title(r'Sintesi [$\it{The \ Cranberries\ -\ Zombie}$]')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()
plt.show()


#---#
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_filtered[:len(fft_filtered)//2])**2, color='green') #filtro basso
axs[0].set_title(r'Potenza spettrale [$\it{The \ Cranberries\ -\ Zombie}$], Strumento: basso ')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')
axs[0].legend(['Potenza spettrale'], fontsize=10)

axs[1].plot(new_data, alpha=0.7, color='green', label='Segnale originale [basso e chitarra]') 
axs[1].plot(anti_fft, color='purple', label='Strumento: basso')
axs[1].set_title(r'Sintesi segnale filtrato [$\it{The \ Cranberries\ -\ Zombie}$]')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()
plt.show()

#----#
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_filtered1[:len(fft_filtered1)//2])**2, color='green') #filtro chitarra
axs[0].set_title(r'Potenza spettrale [$\it{The \ Cranberries\ -\ Zombie}$], Strumento: chitarra')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')
axs[0].legend(['Potenza spettrale'], fontsize=10)

axs[1].plot(new_data, alpha=0.7, color='green', label='Segnale originale [basso e chitarra]') 
axs[1].plot(anti_fft1, color='purple', label='Strumento: chitarra')
axs[1].set_title(r'Sintesi segnale filtrato [$\it{The \ Cranberries\ -\ Zombie}$]')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()
plt.show()

# Salvataggio audio
sf.write('/home/gabriele/Laboratorio3/garage_band/Suoni creati/new_primo(filt0).wav', np.real(anti_fft), samplerate, subtype='FLOAT')
sf.write('/home/gabriele/Laboratorio3/garage_band/Suoni creati/new_primo(filt1).wav', np.real(anti_fft1), samplerate, subtype='FLOAT')
