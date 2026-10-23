# Laboratorio di Elettronica 2024–2025

**Analisi spettrale, filtraggio e ricostruzione di segnali audio con Python.**

Questo repository raccoglie gli script e i file audio sviluppati per il progetto di Laboratorio di Elettronica dell'anno accademico 2024–2025.

L'obiettivo è studiare la composizione in frequenza di segnali e registrazioni musicali tramite **trasformata di Fourier (FFT)**, applicare filtri nel dominio delle frequenze e ricostruire i segnali risultanti tramite trasformata inversa (IFFT).

Il lavoro è suddiviso in tre parti, con un percorso che va dall'analisi di singoli suoni fino al filtraggio di brani musicali.

## Contenuti del progetto

### Parte 1 — Analisi di segnali e singoli suoni

La cartella `Parte1/` contiene due gruppi di esperimenti.

**Analisi di tre dataset numerici**

Lo script `analisi_suoni.py` esegue:
- Lettura di dati sperimentali.
- Calcolo della trasformata di Fourier.
- Analisi delle componenti spettrali.
- Applicazione di filtri in frequenza.
- Ricostruzione e visualizzazione dei segnali.

**Analisi di registrazioni audio**

Gli script `diapason.py`, `distorta.py`, `pulita_semplice.py`, `pulita_media.py` e `pulita_difficile.py` analizzano registrazioni differenti, individuano i principali picchi spettrali e confrontano diverse strategie di filtraggio.

Il file `modulo.py` contiene un'implementazione didattica della trasformata discreta inversa di Fourier (IDFT).

I file WAV già elaborati si trovano nella cartella `Parte1/File audio/`.

### Parte 2 — Filtraggio di estratti musicali

La cartella `Parte2/` contiene due script:

- `pulita_pezzo.py`
- `distorta_pezzo.py`

Questa fase estende l'analisi a estratti audio puliti e distorti.

Gli script applicano filtri che selezionano progressivamente uno o più intervalli di frequenza, permettendo di osservare come cambiano lo spettro e il segnale ricostruito.

I risultati sono disponibili nella cartella `Parte2/File Audio/`.

### Parte 3 — Separazione approssimata degli strumenti

La cartella `Parte3/` applica il filtraggio in frequenza a due brani musicali:

- **The Cranberries — Zombie**, tramite `Cranberries.py`.
- **Nirvana — Smells Like Teen Spirit**, tramite `Nirvana_pezzo.py`.

Gli script suddividono il contenuto spettrale in due bande principali:

| Banda di frequenza | Strumento associato |
|---|---|
| 0–500 Hz | Basso |
| 500–5000 Hz | Chitarra |

Questa procedura permette una **separazione approssimata basata sulle frequenze**, non un isolamento perfetto degli strumenti, poiché le rispettive componenti spettrali possono sovrapporsi.

I risultati audio sono disponibili nella cartella `Parte3/File Audio/`.

## Tecnologie utilizzate

| Tecnologia | Utilizzo |
|---|---|
| Python 3 | Sviluppo degli script |
| NumPy | Elaborazione numerica dei segnali |
| SciPy | Trasformate FFT e IFFT |
| Matplotlib | Visualizzazione dei segnali e degli spettri |
| SoundFile | Lettura e scrittura dei file WAV |
| Pandas | Caricamento e gestione dei dataset |

## Struttura del repository

```text
Laboratorio-elettronica-24-25/
│
├── Parte1/
│   ├── analisi_suoni.py
│   ├── diapason.py
│   ├── distorta.py
│   ├── pulita_semplice.py
│   ├── pulita_media.py
│   ├── pulita_difficile.py
│   ├── modulo.py
│   └── File audio/
│
├── Parte2/
│   ├── pulita_pezzo.py
│   ├── distorta_pezzo.py
│   └── File Audio/
│
├── Parte3/
│   ├── Cranberries.py
│   ├── Nirvana_pezzo.py
│   └── File Audio/
│
└── README.md
```

## Installazione

Clonare il repository:

```bash
git clone https://github.com/GabrieleVirdis/Laboratorio-elettronica-24-25.git
```

Accedere alla cartella:

```bash
cd Laboratorio-elettronica-24-25
```

Installare le librerie necessarie:

```bash
python -m pip install numpy scipy matplotlib soundfile pandas
```

## Esecuzione

Ogni script rappresenta un'analisi indipendente.

Per analizzare i dataset numerici:

```bash
python Parte1/analisi_suoni.py
```

Per analizzare una registrazione audio:

```bash
python Parte1/distorta.py
```

Per eseguire il filtraggio di un estratto musicale:

```bash
python Parte2/pulita_pezzo.py
```

Per eseguire la separazione in bande di frequenza:

```bash
python Parte3/Nirvana_pezzo.py
```

**Nota:** prima dell'esecuzione degli script che utilizzano registrazioni WAV, è necessario modificare i percorsi di input e output presenti nel codice, adattandoli alla posizione effettiva dei file sul proprio computer.

I file audio sorgente non sono tutti inclusi nel repository. Sono invece disponibili numerosi WAV già elaborati.

Lo script `analisi_suoni.py` richiede una connessione a Internet per scaricare i dataset dal server esterno utilizzato nel progetto.

## Metodologia

L'elaborazione dei segnali segue generalmente sei passaggi:

1. **Acquisizione:** lettura del segnale campionato o del file audio.
2. **Trasformata di Fourier:** conversione del segnale dal dominio temporale a quello delle frequenze.
3. **Analisi spettrale:** individuazione delle principali componenti e visualizzazione della potenza spettrale.
4. **Filtraggio:** selezione degli intervalli di frequenza mediante maschere.
5. **Ricostruzione:** applicazione della trasformata inversa per ottenere il segnale filtrato.
6. **Esportazione:** salvataggio dei risultati in formato WAV, dove previsto.

Questo procedimento permette di analizzare il contributo delle diverse componenti spettrali alla forma e al contenuto sonoro dei segnali.

## Note tecniche

Gli script sono stati sviluppati per finalità didattiche e presentano alcune limitazioni.

In `Parte1/diapason.py` viene utilizzata la funzione `idft()` senza importarla. È possibile aggiungere:

```python
from modulo import idft
```

In alternativa, si può utilizzare `fft.ifft()` per una ricostruzione più efficiente.

Le maschere degli script audio selezionano generalmente soltanto le frequenze positive. Per preservare correttamente un segnale reale, il filtraggio dovrebbe considerare anche le corrispondenti frequenze negative.

Gli intervalli di filtraggio sono inoltre specifici dei segnali analizzati e non costituiscono filtri universali applicabili a qualsiasi registrazione.

## Autore

**[Gabriele Virdis](https://github.com/GabrieleVirdis)**

*Progetto didattico — Anno accademico 2024–2025.*
