import numpy as np
import scipy
import scipy.signal
import scipy.fft
import scipy.integrate
from math import cos, sin, pi
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.optimize import linprog
from math import pi


# sound -- to process
# [min_freq, max_freq)
# Really returns twice the filter.
# Most useful for real data `sound`
# (can apply on complex sound, but divide by 2 at the end.)
# Return value is numpy array with dtype=float32
def bandwidth_filter(center_freq, sound, num_bands: int, sample_width: int, SR: int):
    # TODO: adjust the filter to allow smooth freq thresholds.
    # Function which approximates rect but band limited Fourier series
    sinc = LP_filter(center_freq, num_bands, 0.25, SR, sample_width)
    np_sinc = np.array(sinc, dtype=np.complex64)
    conv = scipy.signal.oaconvolve(np_sinc, sound)
    half = len(sinc) // 2
    res = conv[half:-half]
    assert len(res) == len(sound)
    return res


def normalize_freq_SR(freq, SR: int):
    if is_integer(freq):
        center_freq = round(freq)
    elif is_integer(2 * freq):
        center_freq = round(2 * freq)
        SR *= 2
    else:
        print("unphasing with non-integer frequency")
    return (freq, SR)


def unphase_wave(data, freq, SR: int):
    (freq, SR) = normalize_freq_SR(freq, SR)
    # Faster than cos + 1j * sin
    phases = np.multiply(
        np.mod(freq * np.arange(0, len(data)), SR),
        -2j * pi / SR,
        dtype=np.complex64
    )
    unphased = np.multiply(data, np.exp(phases))
    return unphased


def is_integer(x):
    return abs(x - round(x)) < 1E-10


# Useful for frequency shift of filters
def unphase_filter(data, freq, SR: int, dtype=np.complex64):
    (freq, SR) = normalize_freq_SR(freq, SR)
    phases = np.multiply(
        np.mod(freq * np.arange(-len(data) // 2 + 1, len(data) // 2 + 1), SR),
        -2j * pi / SR,
        dtype=np.complex128
    )
    unphased = np.multiply(data, np.exp(phases))
    return np.array(unphased, dtype=dtype)


@lru_cache
def get_remez(freq_threshold, interpolation_width, length):
    # with some hardcoded default params
    assert length % 2 == 1
    res = scipy.signal.remez(length,
                             [0, freq_threshold * (1 - interpolation_width), freq_threshold * (1 + interpolation_width),
                              0.5], [1, 0], fs=1.0)
    res[length // 2] = 2 * freq_threshold
    return res


def remez_filter(min_freq, max_freq, SR: int, length: int):
    # In the LP do not foget to set the bias to be what it should be.
    freq = (max_freq - min_freq) / 2.0
    res = get_remez(freq / SR, 0.25, length)
    res = np.array(res, dtype=np.complex128)
    return unphase_filter(res, -(min_freq + max_freq) / 2, SR)


# a+1 points
def arange(s, e, a, include_sp=True, include_ep=True):
    if include_sp: yield s
    inv = (e - s) / a
    for i in range(1, a):
        yield s + i * inv
    if include_ep: yield e


def fast_LP_filter(num_bands: int, interpolation_width, length: int):
    res = [0.0] * length
    remez = get_remez(1 / (2 * num_bands), interpolation_width, length)
    middle = length // 2
    for i in range(length):
        if (i - middle) % num_bands != 0 or i == middle:
            res[i] = remez[i]
    return res


# TODO: eliminate SR argument. Redundant.
@lru_cache
def base_LP_filter(num_bands: int, interpolation_width, SR: int, length: int, fast_impl=True):
    assert length % 2 == 1
    if fast_impl:
        # Sometimes Remez fails and we can handle this by the next LP code.
        # However, it is very slow and common cases should be cached.
        return fast_LP_filter(num_bands, interpolation_width, length)

    # This is the value of tap[0], in order for all bands to sum to 1.
    middle_tap_idx = length // 2
    middle_tap_value = 1 / num_bands
    # all other taps <= max_abs
    max_abs = middle_tap_value / 2
    # We assume filter is symmetric, and formulate LP only on half the taps
    # We use only the taps that are allowed and will not introduce systematic errors.
    # NOTE: maybe LP_filter can be replaced by remez where unusable taps are zeroed out
    usable_taps = [tap for tap in range(1, length // 2 + 1) if tap % num_bands != 0]
    # interpolation_width = percentage of band length devoted to fading out. In range (0,1)
    # interpolation_width = 1: filter looks like a triangle with base 2 * SR / num_bands, with a single peak (Fejer).
    # interpolation_width = 0: filter looks like a rectangle of width SR / num_bands (sinc).
    assert 0 < interpolation_width < 1
    # # in frequency domain
    num_samples_pass_band = 2 * (length - 1)
    num_samples_interpol_band = 1 * (length - 1)
    num_samples_mask_band = 4 * (length - 1)
    interpol_mid = SR / (2 * num_bands)
    interpol_start = interpol_mid * (1 - interpolation_width)
    interpol_end = interpol_mid * (1 + interpolation_width)
    pass_pts = list(arange(0, interpol_start, num_samples_pass_band))
    interpol_pts = list(arange(interpol_start, interpol_mid, num_samples_interpol_band))
    mask_pts = list(arange(interpol_end, SR / 2, num_samples_mask_band))
    A = []
    B = []
    C = [0] * len(usable_taps) + [1]

    def add_closeness_constraint(coefs, value):
        value = value / 2 + max_abs * sum(coefs)
        A.append(coefs + [-1])
        B.append(value)
        A.append([-c for c in coefs] + [-1])
        B.append(-value)

    def FT(x):
        return [cos(2 * pi * tap * x / SR) for tap in usable_taps]

    for (arr, val) in ((pass_pts, 1), (mask_pts, 0)):
        for x in arr:
            add_closeness_constraint(FT(x), val - 1 * middle_tap_value)
    for x in interpol_pts:
        y = 2 * interpol_mid - x
        FTxy = [a + b for a, b in zip(FT(x), FT(y))]
        add_closeness_constraint(FTxy, 1 - 2 * middle_tap_value)
    # linprog assumes all variables >= 0, so we shifted all variables by max_abs
    ans = linprog(C, A, B)
    if ans.status != 0:
        print("Error code in linprog in LP_filter. error_code=", ans.status, ". Quality=", ans.x[-1])
    tap_values = ans.x[:-1]
    assert len(tap_values) == len(usable_taps), (len(tap_values), len(usable_taps))
    # not forget inserting middle_tap, and their symmetric.
    res = [0.0] * length
    res[middle_tap_idx] = middle_tap_value
    for tap, value in zip(usable_taps, tap_values):
        value = float(value)
        res[middle_tap_idx + tap] = value - max_abs
        res[middle_tap_idx - tap] = value - max_abs
    return res


def LP_filter(center_freq, num_bands: int, interpolation_width, SR: int, length: int):
    # Note num_bands refers to number of bands required for tiling all [-SR/2, SR/2].
    # In practice, we often use only half the number of bands, to tile only [0, SR/2],
    # and the real part is exactly the sound wave.
    return unphase_filter(base_LP_filter(num_bands, interpolation_width, SR, length), -center_freq, SR)


def plot_fft(filter, name):
    L = 2 ** 18
    B = [0] * L
    assert len(filter) % 2 == 1
    B[:len(filter) // 2 + 1] = filter[len(filter) // 2:]
    B[-len(filter) // 2:] = filter[:len(filter) // 2]
    C = np.fft.fft(B)
    C = list(C[:len(C) // 20]) + list(C[-len(C) // 20:])
    xs = list(range(len(C)))
    plt.plot(xs, np.abs(C), label=name, alpha=0.4)


if __name__ == "__main__":
    # visualizing the filter.
    SR = 22050
    num_bands = 75
    filter_size = 513
    min_freq = -SR / (2 * num_bands)
    max_freq = SR / (2 * num_bands)
    # A1 = get_sinc(min_freq, max_freq, SR, 257)
    A2 = LP_filter(0, num_bands, 0.25, SR, filter_size)
    A3 = remez_filter(min_freq, max_freq, SR, filter_size)
    # A3 = LSQ_filter(min_freq, max_freq, SR, len(A1))
    # plot_fft(A1, "sindiff")
    plot_fft(A2, "LP_filter")
    plot_fft(A3, "remez")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.legend()
    plt.show()

    assert SR % (2 * num_bands) == 0
    smsm = np.array([0] * filter_size, dtype=np.float32)
    for i in range(0, SR, round(max_freq)):
        # smsm = np.add(smsm, get_sinc(i, i + max_freq, SR, len(smsm)))
        # smsm = np.add(smsm, remez_filter(i-max_freq / 2, i+max_freq / 2, SR, len(smsm)))
        smsm = np.add(smsm, LP_filter(i, num_bands, 0.25, SR, len(smsm)))
    smsm[len(smsm) // 2] -= 2
    # numerical errors are eliminated once unphase_filter returns complex128 instead of complex64
    print("total error", np.abs(smsm).sum())
    for i in range(len(smsm)):
        if np.abs(smsm[i]) > 1E-4:
            print(i, smsm[i])
