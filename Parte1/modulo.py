import numpy as np

def idft(X):
    N = len(X)                                # Numero totale di coefficienti
    indici = np.arange(0, N)             # Indici dei campioni da calcolare
    sintesi = np.empty(len(indices), dtype=complex)  # Array per salvare i risultati
    
    # Calcolo dell'IDFT 
    k = np.arange(N)                          # Indici per i coefficienti in frequenza
    for i in range(len(indici)):
        n = indici[i]
        ang = 2 * np.pi * n * k / N
        sintesi[i] = np.sum(X * (np.cos(ang) + 1j * np.sin(ang)))
    
    # Normalizza dividendo per N
    return sintesi / N

