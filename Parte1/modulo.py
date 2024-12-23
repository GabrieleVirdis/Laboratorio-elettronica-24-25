import numpy as np

def idft(fft_coeffs):
    """
    Calcola l'antitrasformata discreta di Fourier (IDFT) senza l'uso di un ciclo for,
    utilizzando np.cos e np.sin.

    Parametri:
        fft_coeffs (np.array): Coefficienti di Fourier.

    Ritorna:
        np.array: Il segnale ricostruito.
    """
    N = len(fft_coeffs)
    n = np.arange(N)  # Indici temporali
    k = np.arange(N)  # Indici delle frequenze

    # Calcolo della matrice degli angoli
    angles = 2 * np.pi * n * k / N

    # Calcolo della matrice esponenziale usando np.cos e np.sin
    exp_matrix = np.cos(angles) + 1j * np.sin(angles)

    # Prodotto matrice-vettore per ottenere l'antitrasformata
    result = np.dot(exp_matrix, fft_coeffs)

    # Normalizzazione del risultato
    return result / N
