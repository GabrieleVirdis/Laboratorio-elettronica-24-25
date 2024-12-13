# Import Moduli
import numpy as np
from scipy import fft, constants
import matplotlib.pyplot as plt
import soundfile as sf

# Leggi il file audio (diapason.wav)
data, samplerate = sf.read('/home/gabriele/Downloads/pulita_semplice(3).wav')
print('-------------------------------')
print('Samplerate:', samplerate)
print('Array:', data)
print('Dimensioni Array:', len(data))
print('-------------------------------')

# Creazione nuovo file array (new_diapason.wav)
sf.write('/home/gabriele/Downloads/new_pulita_semplice.wav', data, samplerate)
new_data, new_samplerate = sf.read('/home/gabriele/Downloads/new_pulita_semplice.wav')

# solo un canale
if len(new_data.shape) > 1:
    new_data = new_data[:, 0] 

# Trasformata di Fourier
fft_coeffs = fft.fft(new_data)
diff_temp = 1 / new_samplerate
freqs =  fft.fftfreq(len(new_data), diff_temp)

'''
# Filtraggi: Maschere 
fft_filtered = fft_coeffs.copy() #picco centrale
mask = np.absolute(fft_coeffs)**2 < (1.2) * 10 ** 8 #picco centrale
fft_filtered[mask] = 0 #picco centrale

fft_filtered1 = fft_coeffs.copy() # due picchi principali (solo t. centrale)
mask1 = ((freqs >= 329.25) & (freqs <= 329.60)) | ((freqs >= 551.80) & (freqs <= 552.00)) # due picchi principali (solo t. centrale)
fft_filtered1[~mask1] = 0 # due picchi principali (solo t. centrale)

fft_filtered2 = fft_coeffs.copy() # tutti i picchi principali (solo t. centrale)
mask2 = ((freqs >= 329.25) & (freqs <= 329.60)) | ((freqs >= 551.80) & (freqs <= 552.00)) | ((freqs >= 662.00) & (freqs <= 662.30)) | ((freqs >= 1543.20) & (freqs <= 1543.60)) | ((freqs >= 2320.00) & (freqs <= 2320.60)) # tutti i picchi principali (solo t. centrale)
fft_filtered2[~mask2] = 0 # tutti i picchi principali (solo t. centrale)

fft_filtered3 = fft_coeffs.copy() # tutti i picchi principali (+2 termini)
mask3 = ((freqs >= 327) & (freqs <= 332)) | ((freqs >= 549.80) & (freqs <= 554.00)) | ((freqs >= 660.00) & (freqs <= 664)) | ((freqs >= 1542.00) & (freqs <= 1545.00)) | ((freqs >= 2318.00) & (freqs <= 2322.00))  # tutti i picchi principali (+ 2 termini)
fft_filtered3[~mask3] = 0 # tutti i picchi principali (+ 2 termini)                        


# Antitrasformata di fourier per segnale filtrato
anti_fft_o = fft.ifft(fft_coeffs)
anti_fft = fft.ifft(fft_filtered)
anti_fft1 = fft.ifft(fft_filtered1)
anti_fft2 = fft.ifft(fft_filtered2)
anti_fft3 = fft.ifft(fft_filtered3)

'''

# Plot dati
fig, axs = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')

# Segnale originale
axs[0, 0].plot(new_data, color='green')
axs[0, 0].set_xlabel('Tempo [s]')
axs[0, 0].set_ylabel('Ampiezza')
axs[0, 0].set_title('Segnale originale (diapason)')
axs[0, 0].legend(['Segnale originale'], fontsize=10)

# Coefficienti di Fourier (parte reale)
axs[0, 1].plot(freqs[:len(freqs)//2], np.real(fft_coeffs[:len(fft_coeffs)//2]), color='yellow')
axs[0, 1].set_title('Parte reale dei coefficienti di Fourier')
axs[0, 1].set_xlabel('Frequenza [Hz]')
axs[0, 1].set_ylabel(r'Re$(X_k)$')
axs[0, 1].legend(['Parte reale FFT'], fontsize=10)

# Coefficienti di Fourier (parte immaginaria)
axs[1, 0].plot(freqs[:len(freqs)//2], np.imag(fft_coeffs[:len(fft_coeffs)//2]), color='orange')
axs[1, 0].set_title('Parte immaginaria dei coefficienti di Fourier')
axs[1, 0].set_xlabel('Frequenza [Hz]')
axs[1, 0].set_ylabel(r'Im$(X_k)$')
axs[1, 0].legend(['Parte immaginaria FFT'], fontsize=10)


# Potenza spettrale e confronto segnali filtrati e originale
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_coeffs[:len(fft_coeffs)//2])**2, color='green') #originale
axs[0].set_title('Potenza spettrale originale (diapason)')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')

#axs[1].plot(anti_fft_o, color='green', label='Segnale originale') #ricostruzione segnale originale
axs[1].set_title('Diapason originale ricostruita')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()
plt.show()

'''
#---#
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_filtered[:len(fft_filtered)//2])**2, color='green') #filtro tranne del picco principale
axs[0].set_title('Potenza spettrale (diapason), picco principale (termine centrale) ')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')
axs[0].legend(['Potenza spettrale'], fontsize=10)

axs[1].plot(new_data, alpha=0.7, color='green', label='Segnale originale') #filtro tranne del picco principale
axs[1].plot(anti_fft, color='purple', label='Segnale filtrato $P < 3.5$')
axs[1].set_title('Segnale ricostruito (filtrato)')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()
plt.show()
#----#
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_filtered1[:len(fft_filtered1)//2])**2, color='green') #filtro tranne dei primi due picchi principali (solo il termine centrale)
axs[0].set_title('Potenza spettrale (diapason), 2 picchi principali (termine centrale)')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')
axs[0].legend(['Potenza spettrale'], fontsize=10)

axs[1].plot(new_data, alpha=0.7, color='green', label='Segnale originale') #filtro tranne picchi principali (solo il termine centrale)
axs[1].plot(anti_fft1, color='purple', label='Segnale filtrato')
axs[1].set_title('Diapason ricostruita filtrata')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()
plt.show()
#---#
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_filtered2[:len(fft_filtered2)//2])**2, color='green') #filtro tranne picchi principali (solo il termine centrale)
axs[0].set_title('Potenza spettrale (diapason), picchi principali (termine centrale)')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')
axs[0].legend(['Potenza spettrale'], fontsize=10)

axs[1].plot( new_data, alpha=0.7, color='green', label='Segnale originale') #filtro tranne picchi principali (solo il termine centrale)
axs[1].plot( anti_fft2, color='purple', label='Segnale filtrato')
axs[1].set_title('Diapason  ricostruita (filtrata)')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()
plt.show()

#---# 
fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

axs[0].plot(freqs[:len(freqs)//2], np.abs(fft_filtered3[:len(fft_filtered3)//2])**2, color='green') #filtro dei picchi principali + 2 termini per lato oltre quello centrale 
axs[0].set_title('Potenza spettrale (distorta), picchi principali (termine centrale + 2)')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel(r'$|X_k|^2$')
axs[0].legend(['Potenza spettrale'], fontsize=10)

axs[1].plot(new_data, alpha=0.7, color='green', label='Segnale originale') #filtro dei picchi principali + 2 termini per lato oltre quello centrale
axs[1].plot(anti_fft3, color='purple', label='Segnale filtrato')
axs[1].set_title('Diapason ricostruita (filtrata)')
axs[1].set_xlabel('Tempo')
axs[1].set_ylabel('Ampiezza')
axs[1].legend()

insets = [
    [327, 332],  # Zoom su primo picco
    [549.80, 554.00],  # Zoom su secondo picco
    [660.00, 664.00],  # Zoom su terzo picco
    [1542.00, 1545.00],  # Zoom su quarto picco
    [2318.00, 2322.00],  # Zoom su quinto picco
]

lim_y = [
    [0, 2 * 10**8],
    [0, 1.3 * 10**8],
    [0, 0.8 * 10**8],
    [0, 0.3 * 10**8],
    [0, 0.1 * 10**8],
]
for i, (start_freq, end_freq) in enumerate(insets):
    # Posizionamento degli insetti in alto a destra con maggiore distanza
    ins_ax = axs[0].inset_axes([0.45 + (i % 2) * 0.25, 0.75 - (i // 2) * 0.25, 0.15, 0.20])  # [x, y, width, height]
    ins_ax.plot(freqs[:len(freqs)//2], np.abs(fft_filtered3[:len(fft_filtered3)//2])**2, color='green')
    ins_ax.set_ylim(lim_y[i])
    ins_ax.set_xlim(start_freq, end_freq)

plt.show()

# Salvataggio audio
sf.write('/home/gabriele/Laboratorio3/garage_band/Suoni creati/new_pulita_semplice(filt0).wav', np.real(anti_fft), samplerate, subtype='FLOAT')
sf.write('/home/gabriele/Laboratorio3/garage_band/Suoni creati/new_pulita_semplice(filt1).wav', np.real(anti_fft1), samplerate, subtype='FLOAT')
sf.write('/home/gabriele/Laboratorio3/garage_band/Suoni creati/new_pulita_semplice(filt2).wav', np.real(anti_fft2), samplerate, subtype='FLOAT')
sf.write('/home/gabriele/Laboratorio3/garage_band/Suoni creati/new_pulita_semplice(filt3).wav', np.real(anti_fft3), samplerate, subtype='FLOAT')
'''
