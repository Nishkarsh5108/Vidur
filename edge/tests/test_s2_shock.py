"""S2's input contract (docs/ml-models-spec.md §7): the easy-to-get-wrong parts."""

import numpy as np

from edge.runners import s2_shock


def _window(seed=0):
    return np.random.default_rng(seed).normal(0.0, 0.5, s2_shock.WINDOW)


def test_vibration_is_channels_last_and_interleaved():
    az = _window()
    vibration, _ = s2_shock.preprocess(az, 12.0)
    sig = np.diff(az, prepend=az[0])
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    assert vibration.shape == (1, 128, 2)
    flat = vibration.ravel()
    # The buffer is c0[0], c1[0], c0[1], c1[1], ... (not all of c0 followed by all of c1).
    np.testing.assert_allclose(flat[0::2], np.abs(sig), rtol=1e-6)
    np.testing.assert_allclose(flat[1::2], np.gradient(sig), rtol=1e-5, atol=1e-6)


def test_context_uses_real_scaler_and_rms_without_epsilon():
    az = _window(1)
    _, context = s2_shock.preprocess(az, 12.464910954632682)
    sig = np.diff(az, prepend=az[0])
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    rms = np.sqrt(np.mean(sig ** 2))                     # training: no epsilon
    expected_rms_feature = (rms - 0.9999906116945316) / 6.002955638863807e-06
    assert context.shape == (1, 4)
    assert context[0, 1] == 0.0                          # speed equal to the scaler mean
    np.testing.assert_allclose(context[0, 2], expected_rms_feature, rtol=1e-4)
    # vid.py's rms + 1e-6 would shift this feature by ~0.17 scaler units: big enough to matter.
    assert abs(((rms + 1e-6) - 0.9999906116945316) / 6.002955638863807e-06 - context[0, 2]) > 0.1


def test_gravity_offset_cancels():
    az = _window(2)
    a, ca = s2_shock.preprocess(az, 10.0)
    b, cb = s2_shock.preprocess(az - 9.81, 10.0)
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(ca, cb, atol=1e-3)


def test_confidence_gate():
    assert s2_shock.gate(np.array([0.05, 0.05, 0.1, 0.8])) == 0    # unsure Big Pothole -> Excellent
    assert s2_shock.gate(np.array([0.0, 0.0, 0.1, 0.9])) == 3
    assert s2_shock.gate(np.array([0.1, 0.6, 0.2, 0.1])) == 1      # classes 0-1 are never gated
