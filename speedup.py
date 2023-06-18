from math import pi
from functools import lru_cache
import numpy as np
import scipy.signal
from utils import norm_squared
import subband

NUM_BANDS = 14
DEFAULT_WINDOW_SIZE_MS = 50
SUBBAND_FILTER_SIZE_MS = 10


def np_arange(start, step, length):
    return start + np.arange(length) * step


@lru_cache
def continuous_average(ln):
    if ln < 2:
        A = [0.5] * ln
        B = [0.5] * ln
    else:
        # B = np.arange(0, ln) * (1/(ln - 1))
        # A = 1-B
        p = np_arange(0, pi / (2 * (ln - 1)), ln)
        A = np.cos(p)
        A = np.multiply(A, A)
        B = 1 - A
    return [np.array(C, dtype=np.float32) for C in [A, B]]


def speedup_naive(data, SR: int):
    block = (SR * DEFAULT_WINDOW_SIZE_MS) // 1000
    res = [0] * (len(data) // 2)
    i = 0
    while i < len(data):
        j = min(i + 2 * block, len(data))
        k = (i + j) // 2
        ln = k - i
        A, B = continuous_average(ln)
        D1, D2 = data[i:k], data[k:k + ln]
        res[i // 2: j // 2] = np.add(np.multiply(D1, A), np.multiply(D2, B))
        i = j
    return res


def amdf_score(data, index, shift):
    return np.sum(np.abs(data[index:index + shift] - data[index + shift:index + 2 * shift])) / shift


def speedup_pitch_aware(data, SR: int, verbose=False):
    # does not interfere much with phases
    max_shift = SR * DEFAULT_WINDOW_SIZE_MS // 1000
    min_shift = max_shift // 3
    res = [0] * (len(data) // 2)
    i = 0
    prev_shift = max_shift
    while i < len(data):
        if i + 2 * max_shift > len(data): break
        C = np.real(
            scipy.signal.oaconvolve(data[i:i + max_shift], np.conj(data[i + 2 * max_shift - 1:i + min_shift - 1:-1]))[
            max_shift - 1:1 - max_shift])
        assert len(C) == max_shift - min_shift + 1
        # TODO: choose best shift according to AMDF like libsonic?
        best_shift = max_shift - np.argmax(C)
        if amdf_score(data, i, prev_shift) < amdf_score(data, i, best_shift):
            best_shift = prev_shift
        prev_shift = best_shift
        # TODO: locally optimize best_shift a little bit according to shift_score?
        k = i + best_shift
        j = k + best_shift
        A, B = continuous_average(best_shift)
        D1, D2 = data[i:k], data[k:j]
        res[i // 2: j // 2] = np.add(np.multiply(D1, A), np.multiply(D2, B))
        i = j
    if i < len(data):
        res[i // 2:] = speedup_naive(data[i:], SR)
    return res


##################################
##################################
###### Code for algorithm 2 ######
##################################
##################################

def fit_in_bounds(M, min=-1, max=1):
    # TODO: smoother transformation
    return np.minimum(np.maximum(M, min), max)


def subbands_iterator(data, sample_rate, num_bands):
    sample_width = sample_rate * SUBBAND_FILTER_SIZE_MS // 1000
    sample_width += 1 - sample_width % 2
    half_bands = num_bands // 2 + 1
    window = sample_rate / num_bands
    for freq_idx in range(half_bands):
        freq = freq_idx * window
        multiplicity = 2 - (freq_idx == 0)
        if num_bands % 2 == 0:
            multiplicity -= (freq_idx + 1 == half_bands)
        filtered = multiplicity * subband.bandwidth_filter(freq, data, num_bands, sample_width, sample_rate)
        yield freq, filtered


# Uses PSOLA-like algorithm on (unphased) subbands, and sum them up.
def algorithm2(data, sample_rate: int, num_bands=NUM_BANDS, fallback_to_twice_num_bands=True):
    # TODO: for some regions of audio different number of bands are preferable.
    #       We may understand where it happens and choose the bands adaptively
    # sometimes Remez filter raises Exceptions. We try doubling num_bands in this case.
    try:
        res = np.zeros(len(data) // 2, dtype=np.float32)
        # TODO: It is possible to handle all subbands simultaneously, and get ~10x speedup while still
        #       getting the exact same result. It is so because the algorithm is simple in frequency domain.
        for base_freq, filtered_data in subbands_iterator(data, sample_rate, num_bands):
            unphased = subband.unphase_wave(filtered_data, base_freq, sample_rate)
            spedup = speedup_pitch_aware(unphased, sample_rate)
            synthesized = subband.unphase_wave(spedup, -base_freq, sample_rate)
            res = np.add(res, np.real(synthesized))
        return fit_in_bounds(res)
    except:
        if not fallback_to_twice_num_bands: raise
    return algorithm2(data, sample_rate, 2 * num_bands, False)


##################################
##################################
###### Code for algorithm 1 ######
##### which is perhaps worse #####
##################################
##################################

DECOMPOSITION_MIN_FFT_SIZE = 2 ** 13
DECOMPOSITION_NUM_ITERATIONS = 5
DECOMPOSITION_ITERATION_COEF = 1.0 / DECOMPOSITION_NUM_ITERATIONS
DECOMPOSITION_MAX_QUOTIENT = 10
DECOMPOSITION_FREQ_GLUING = 0.1
EPS = 1E-9


def break_to_indices(a, b, n):
    if a < 0:
        yield (a % n, n)
        a = 0
    if b > n:
        yield (0, b % n)
        b = n
    yield (a, b)


# arr[i] is a maximum if argmax(arr[i-wing_len:i+wing_len+1]) = i
def local_maximas(arr, wing_len):
    # A practically fast impl if arr is numpy array, and wing_len is large
    arr_copy = np.array(arr)
    res = []
    while True:
        i = np.argmax(arr_copy)
        if arr_copy[i] == 0: break
        is_max = True
        for min, max in break_to_indices(i - wing_len, i + wing_len + 1, len(arr)):
            if np.max(arr[min:max]) > arr[i]: is_max = False
            arr_copy[min:max] = 0
        if is_max: res.append(i)
    return res


# Outputs array of frequencies which strongly appear in the (weighted) data.
def strong_signals(data, weights, highlights, freq_dist):
    data = np.multiply(data, weights)
    N = len(data)
    # TODO: perform large FFT only when found a non-highlighted frequency?
    while (N & (N - 1)) or N < DECOMPOSITION_MIN_FFT_SIZE: N += N & (-N)
    # allow user to control precision of frequency
    B = np.zeros(N, dtype=np.float32)
    B[:len(data)] = data
    F = np.fft.fft(B)
    # It did not help to normalize by human auditory perception
    amplitudes = np.abs(F[:len(F) // 2])
    res = []
    # TODO: allow wing_len depend on the frequency?
    max_val = 0
    maximas = local_maximas(amplitudes, max(N // 64, 1))
    min_f, max_f = len(data), 0
    # Do not handle signals with much different magnitudes together
    max_q = DECOMPOSITION_MAX_QUOTIENT
    for i in sorted(maximas, key=lambda j: -amplitudes[j]):
        u, v, w = np.abs(F[i - 1]), np.abs(F[i]), np.abs(F[i + 1])
        if v > max_q * min_f or v < max_f / max_q: continue
        min_f = min(min_f, np.abs(F[i]))
        max_f = max(max_f, np.abs(F[i]))
        # second order approximation
        di = (w - u) / ((4 * v - 2 * u - 2 * w) + EPS)
        ii = i + di
        freq = float(ii * 2 * pi / len(B))
        for highlight in highlights:
            # If the highlight is not too bad, use it.
            # TODO: experiment with allowing to change and not fallback to highlight
            # TODO: epriment with finding good "global" frequencies. Currently we slightly overfit.
            if abs(highlight - freq) < freq_dist:
                freq = highlight
        res.append(freq)
        # a is the guessed amplitude, and c is the phase (associated with freq)
        # u, v, w = F[i - 1], F[i], F[i + 1]
        # ff = v + 0.5 * di * (w - u)
        # a = float(abs(ff) * 2 / sum(weights))
        # c = float(np.angle(ff))
        # res.append([float(a), freq, float(c)])
    return list(set(res))


# a cos(bx + c) for x = 0..length
def synthesize(a, b, c, length):
    return a * np.cos(np_arange(c, b, length))


def get_window(length):
    a = pi / length
    res = synthesize(1, a, (a - pi) / 2, length)
    return res  # np.multiply(res, res)


def real_mod(num, mod):
    I = round(num / mod)
    return num - I * mod


# returns a list of tuples (a, freq, c)
# containing all freq in freqs.
# Should be interpreted as:
#   sum a * cos(freq * x + b) ~ data[x]
def purify_signal(data, sqrt_weight, freqs):
    rows = []
    for freq in freqs:
        rows.append(np.multiply(synthesize(1, freq, 0, len(data)), sqrt_weight))
        rows.append(np.multiply(synthesize(1, freq, -pi / 2, len(data)), sqrt_weight))
    # TODO: most inner products here are easy to compute beforehand, which allows speedup.
    sol = np.linalg.lstsq(np.matrix(rows).transpose(), np.multiply(data, sqrt_weight), rcond=None)[0]
    res = []
    for i in range(len(freqs)):
        re, im = sol[2 * i], sol[2 * i + 1]
        coef = re - 1j * im
        # reduce a: do not be so sure about your results!
        a = np.abs(coef) * DECOMPOSITION_ITERATION_COEF
        c = np.angle(coef)
        res.append((a, freqs[i], c))
    return res


def pure_of_half(new_location, pure_signals):
    res = []
    for (a, b, c) in pure_signals:
        # Note we are yet to solve the issue that phases
        # of similar frequencies may be much different.
        # Doing that would be great.
        c = real_mod(b * new_location, 2 * pi)
        res.append((a, b, c))
    return res


def speedup_pure_decomposition(data, SR: int, iters: int, verbose=False):
    # TODO: have the windows more overlapping. For this we need to change the loc steps and the weights.
    # TODO: maybe the synthesize weights should not be as the weights for finding the pure signals?
    window_size = (SR * DEFAULT_WINDOW_SIZE_MS) // 1000
    window_size += (-window_size) % 4
    weights = get_window(window_size)
    double_weights = get_window(2 * window_size)
    half_weights = get_window(window_size // 2)
    sqrt_weights = np.sqrt(weights)
    res = np.zeros(len(data) // 2, dtype=np.float32)
    if verbose: print(f"{norm_squared(data)=}")
    # Not so great that influenhce is only one directional
    signals = []
    for iteration in range(iters):
        remain = np.array(data)
        for loc in range(0, len(data) - window_size, window_size // 2):
            W = data[loc:loc + window_size]
            # At this point, signals are the previous signals
            signals = strong_signals(W, weights, signals, DECOMPOSITION_FREQ_GLUING)
            if len(signals) > 0:
                pure_signals = purify_signal(W, sqrt_weights, signals)
                half_pure = pure_of_half(loc // 2, pure_signals)
                remain[loc:loc + window_size] -= np.multiply(
                    sum(synthesize(a, b, c, window_size) for a, b, c in pure_signals),
                    weights
                )
                res[loc // 2: (loc + window_size) // 2] += np.multiply(
                    sum(synthesize(a, b, c, window_size // 2) for a, b, c in half_pure),
                    half_weights
                )
        if verbose: print(f"{iteration=}, {norm_squared(remain)=}")
        data = remain
    res += algorithm2(data, SR)
    return res


# Decomposes audio to pure signals, synthesizes the most dominant ones
# with frequency(!) locking, and leaves the rest to algorithm 2.
# Probably algorithm 2 is just better.
def algorithm1(data, sample_rate: int):
    iters = DECOMPOSITION_NUM_ITERATIONS
    # TODO: Split to speech and non speech frequencies and apply different algorithms?
    return fit_in_bounds(speedup_pure_decomposition(data, sample_rate, iters=iters, verbose=False))
