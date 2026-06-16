import numpy as np


def recortar_voz(sig, umbral=0.08):
    sig = np.asarray(sig, dtype=float)

    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)

    if len(sig) == 0:
        return sig

    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    idx = np.where(np.abs(sig) > umbral)[0]

    if len(idx) == 0:
        return sig

    inicio = max(idx[0] - 500, 0)
    fin = min(idx[-1] + 500, len(sig))

    return sig[inicio:fin]


def recortar_vocal(sig):
    sig = recortar_voz(sig)

    if len(sig) == 0:
        return sig

    # Antes era 0.45.
    # Ahora conserva más transición consonante-vocal.
    inicio = int(len(sig) * 0.30)
    return sig[inicio:]


def hamming_window(N):
    if N <= 1:
        return np.ones(N)

    n = np.arange(N)
    return 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))


def preemphasis(sig, alpha=0.97):
    sig = np.asarray(sig, dtype=float)

    if len(sig) == 0:
        return sig

    y = np.zeros_like(sig)
    y[0] = sig[0]

    for n in range(1, len(sig)):
        y[n] = sig[n] - alpha * sig[n - 1]

    return y


def dft_manual(sig, sr, max_freq=5000, step=20):
    sig = np.asarray(sig, dtype=float)

    N = min(len(sig), 4096)

    if N == 0:
        return np.array([]), np.array([])

    inicio = max((len(sig) - N) // 2, 0)
    sig = sig[inicio:inicio + N]

    sig = sig - np.mean(sig)
    sig = sig * hamming_window(N)

    freqs = np.arange(0, max_freq + step, step)
    mag = np.zeros(len(freqs))

    for k, f in enumerate(freqs):
        real = 0.0
        imag = 0.0

        for n in range(N):
            angulo = -2 * np.pi * f * n / sr
            real += sig[n] * np.cos(angulo)
            imag += sig[n] * np.sin(angulo)

        mag[k] = np.sqrt(real**2 + imag**2)

    if np.max(mag) > 0:
        mag = mag / np.max(mag)

    return freqs, mag


def energia_banda(freqs, mag, fmin, fmax):
    idx = np.where((freqs >= fmin) & (freqs < fmax))[0]

    if len(idx) == 0:
        return 0.0

    return float(np.sum(mag[idx] ** 2))


def frecuencia_dominante(freqs, mag):
    idx = np.where(freqs > 50)[0]

    if len(idx) == 0:
        return 0.0

    pos = idx[np.argmax(mag[idx])]
    return float(freqs[pos])


def centroide_espectral(freqs, mag):
    idx = np.where(freqs > 50)[0]

    if len(idx) == 0:
        return 0.0

    f = freqs[idx]
    m = mag[idx]

    den = np.sum(m) + 1e-12
    return float(np.sum(f * m) / den)


def estimate_f0_autocorrelation(sig, sr, min_f=70, max_f=350):
    sig = recortar_vocal(sig)

    if len(sig) == 0:
        return 0.0

    sig = sig - np.mean(sig)

    min_lag = int(sr / max_f)
    max_lag = int(sr / min_f)

    if max_lag >= len(sig):
        max_lag = len(sig) - 1

    best_lag = min_lag
    best_corr = -np.inf

    for lag in range(min_lag, max_lag + 1):
        corr = 0.0

        for n in range(len(sig) - lag):
            corr += sig[n] * sig[n + lag]

        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return float(sr / best_lag)


def extract_formants(sig, sr):
    sig = recortar_vocal(sig)

    if len(sig) == 0:
        return [0.0, 0.0]

    sig = sig - np.mean(sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    sig = preemphasis(sig)

    freqs, mag = dft_manual(sig, sr, max_freq=5000, step=20)

    def maximo_en_banda(fmin, fmax):
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

        if len(idx) == 0:
            return 0.0

        pos = idx[np.argmax(mag[idx])]
        return float(freqs[pos])

    f1 = maximo_en_banda(200, 1000)

    # Antes era 800 a 2500.
    # Se baja a 500 para detectar mejor la vocal /u/.
    f2 = maximo_en_banda(500, 2200)

    return [f1, f2]


def extract_features(sig, sr):
    vocal = recortar_vocal(sig)

    f0 = estimate_f0_autocorrelation(vocal, sr)

    formants = extract_formants(vocal, sr)
    f1 = formants[0]
    f2 = formants[1]

    freqs, mag = dft_manual(vocal, sr, max_freq=5000, step=20)

    e_grave = energia_banda(freqs, mag, 100, 500)
    e_baja = energia_banda(freqs, mag, 200, 800)
    e_media = energia_banda(freqs, mag, 800, 1800)

    fdom = frecuencia_dominante(freqs, mag)
    centroide = centroide_espectral(freqs, mag)

    relacion_e = e_baja / (e_media + 1e-9)
    relacion_f = f1 / (f2 + 1e-9)
    relacion_grave_media = e_grave / (e_media + 1e-9)

    return np.array([
        f0,
        f1,
        f2,
        e_grave,
        e_baja,
        e_media,
        fdom,
        centroide,
        relacion_e,
        relacion_f,
        relacion_grave_media
    ])