import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose

from stoneforge.wavelets.butterworth import Butter_Wavelet
from stoneforge.wavelets.ormsby import getOrmsby, _tcrop
from stoneforge.wavelets.ricker import Ricker_Wavelet


def test_butter_wav_default():
    t, wav = Butter_Wavelet()
    assert isinstance(t, np.ndarray)
    assert isinstance(wav, np.ndarray)
    assert t.shape == wav.shape
    assert np.isfinite(wav).all()
    assert len(t) % 2 == 1
    assert np.argmax(np.abs(wav)) == len(wav) // 2


def test_butter_wav_custom_parameters():
    t, wav = Butter_Wavelet(Freq_low=2.0, Freq_hi=40.0, Samples=81, Dt=2)
    assert isinstance(t, np.ndarray)
    assert isinstance(wav, np.ndarray)
    assert t.shape == wav.shape
    assert len(t) % 2 == 1
    assert np.isfinite(wav).all()
    assert np.max(np.abs(wav)) > 0


def test_butter_wav_even_sample_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        t, wav = Butter_Wavelet(Samples=10, Dt=4)
        assert len(t) == 17
        assert len(w) == 1
        assert 'one sample removed from time axis' in str(w[0].message)
        assert t[0] == -t[-1]
        assert t.shape == wav.shape


def test_ormsby_default():
    t, wav = getOrmsby()
    assert isinstance(t, np.ndarray)
    assert isinstance(wav, np.ndarray)
    assert t.shape == wav.shape
    assert np.allclose(np.max(np.abs(wav)), 1.0)
    assert len(t) % 2 == 1
    assert np.isfinite(wav).all()
    assert np.argmax(np.abs(wav)) == len(wav) // 2


def test_ormsby_custom_frequencies():
    freqs = (2.0, 5.0, 25.0, 30.0)
    t, wav = getOrmsby(f=freqs, Samples=91, Dt=2)
    assert isinstance(t, np.ndarray)
    assert isinstance(wav, np.ndarray)
    assert t.shape == wav.shape
    assert np.allclose(np.max(np.abs(wav)), 1.0)
    assert np.isfinite(wav).all()
    assert np.allclose(np.diff(t), np.diff(t)[0])
    assert t[0] == -t[-1]


def test_ormsby_invalid_frequency_length():
    with pytest.raises(AssertionError):
        getOrmsby(f=(5.0, 15.0, 30.0), Samples=71, Dt=4)


def test_ormsby_even_sample_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        t, wav = getOrmsby(Samples=10, Dt=4)
        assert len(t) == 17
        assert len(w) == 1
        assert 'one sample removed from time axis' in str(w[0].message)
        assert t[0] == -t[-1]


def test_tcrop_odd_length():
    t = np.arange(7) * 0.004
    cropped = _tcrop(t)
    assert cropped.shape == t.shape
    assert np.array_equal(cropped, t)


def test_tcrop_even_length():
    t = np.arange(8) * 0.004
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cropped = _tcrop(t)
        assert len(cropped) == 7
        assert len(w) == 1
        assert 'one sample removed from time axis' in str(w[0].message)


def test_ricker_default():
    t, wav = Ricker_Wavelet()
    assert isinstance(t, np.ndarray)
    assert isinstance(wav, np.ndarray)
    assert t.shape == wav.shape
    assert len(t) % 2 == 1
    assert np.isfinite(wav).all()
    assert np.argmax(np.abs(wav)) == len(wav) // 2


def test_ricker_custom_parameters():
    t, wav = Ricker_Wavelet(Peak_freq=20, Samples=81, Dt=2)
    assert isinstance(t, np.ndarray)
    assert isinstance(wav, np.ndarray)
    assert t.shape == wav.shape
    assert len(t) % 2 == 1
    assert np.isfinite(wav).all()
    assert np.max(np.abs(wav)) > 0


def test_ricker_even_sample_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        t, wav = Ricker_Wavelet(Samples=10, Dt=4)
        assert len(t) == 17
        assert len(w) == 1
        assert 'one sample removed from time axis' in str(w[0].message)
        assert t[0] == -t[-1]


def test_ricker_wavelet_symmetry():
    t, wav = Ricker_Wavelet(Peak_freq=15, Samples=81, Dt=4)
    assert np.allclose(wav, wav[::-1], atol=1e-6)
    assert np.allclose(t, -t[::-1], atol=1e-6)
