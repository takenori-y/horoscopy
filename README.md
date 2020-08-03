horoscopy
=========
A python package for speech signal processing.

Installation
------------
The latest stable release can be installed through PyPI by running
```sh
pip install horoscopy
```
Alternatively,
```sh
git clone https://github.com/takenori-y/horoscopy.git
pip install -e horoscopy
```

Examples
--------

### Mel-cepstral analysis
```python
import horoscopy
import librosa

# Compute STFT from audio.
y, _ = librosa.load('hoge.wav', sr=None)
S = np.abs(librosa.stft(y))

# Estimate mel-cepstral coefficients.
C = horoscopy.stft_to_mcep(S, M=24)
```
