horoscopy
=========
A python package for speech signal processing.

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://takenori-y.github.io/horoscopy/v.0.1.0/index.html)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://takenori-y.github.io/horoscopy/latest/index.html)
[![PyPI](https://img.shields.io/pypi/v/horoscopy.svg)](https://pypi.python.org/pypi/horoscopy)
[![License](https://img.shields.io/pypi/l/horoscopy.svg)](https://github.com/takenori-y/horoscopy/blob/master/LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3971002.svg)](https://doi.org/10.5281/zenodo.3971002)

Documentation
-------------
See [this page](https://takenori-y.github.io/horoscopy/v.0.1.0/index.html) for a reference manual.

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

# Compute STFT of audio.
y, _ = librosa.load('hoge.wav', sr=None)
S = np.abs(librosa.stft(y))
```

Acknowledgements
----------------
This library is inspired by the following open source projects:

- [SPTK](https://github.com/sp-nitech/SPTK)
- [librosa](https://github.com/librosa/librosa)
- [SciPy](https://github.com/scipy/scipy)
