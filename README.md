# TimeStretch2X

Tiny Python 2x audio time stretch (speedup) algorithm: uses PSOLA on subbands of the input.

My subjective judgment is that it sounds better than PSOLA (which is pitch aware. Implementation included -- it computes pitch with autocorrelation using FFT).

The algorithm we use here:

  1. Decompose input sound to several uniformly-spaced subbands.
  2. Apply PSOLA on each subband independently.
  3. Sum the subbands.

See `main.py` for a simple usage example. See samples for a few input / output samples.

The algorithm may easily be extended to arbitrary time stretching (the PSOLA method should be adjusted accordingly).
