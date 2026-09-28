"""S1 pre-processing must match training exactly, and inputs must be bound by name."""

import numpy as np
import pytest

from edge.config import repo_path
from edge.runners import s1_iri


def notebook_context_features(raw_data_window, speed_array):
    """Verbatim from ML_models/IRI/iri_compliant/model.ipynb (extract_context_features)."""
    ax, ay, az, wx, wy, wz = [raw_data_window[:, i] for i in range(6)]
    N = len(az)
    speed_mean = np.mean(speed_array)
    speed_std = np.std(speed_array)
    rms_az = np.sqrt(np.mean(az**2))
    rms_ay = np.sqrt(np.mean(ay**2))
    var_az = np.var(az)
    crest_factor_az = np.max(np.abs(az)) / (rms_az + 1e-6)
    zero_crossings_az = len(np.where(np.diff(np.sign(az)))[0])
    mcr_az = zero_crossings_az / N
    p2p_az = np.max(az) - np.min(az)
    rms_wz = np.sqrt(np.mean(wz**2))
    rms_wy = np.sqrt(np.mean(wy**2))
    mean_abs_ax = np.mean(np.abs(ax))
    fft_vals = np.fft.rfft(az)
    psd = np.abs(fft_vals)**2
    freqs_spatial = np.fft.rfftfreq(N, d=0.25)
    if speed_mean > 0:
        freqs_temporal = freqs_spatial * speed_mean
        band_1_4 = np.sum(psd[(freqs_temporal >= 1.0) & (freqs_temporal <= 4.0)])
        band_4_15 = np.sum(psd[(freqs_temporal > 4.0) & (freqs_temporal <= 15.0)])
        total_energy = np.sum(psd) + 1e-6
        energy_ratio_1_4 = band_1_4 / total_energy
        energy_ratio_4_15 = band_4_15 / total_energy
    else:
        energy_ratio_1_4, energy_ratio_4_15 = 0.0, 0.0
    return np.array([speed_mean, speed_std, rms_az, rms_ay, var_az, crest_factor_az, mcr_az, p2p_az,
                     rms_wz, rms_wy, mean_abs_ax, energy_ratio_1_4, energy_ratio_4_15])


def test_context_features_match_training():
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, (400, 6))
    speed = rng.uniform(5, 25, 400)
    np.testing.assert_allclose(s1_iri.context_features(raw, speed), notebook_context_features(raw, speed))


def test_window_resampling_and_speed_normalisation():
    n = 300
    dist = np.linspace(0, 100, n)
    imu6 = np.ones((n, 6))
    speed = np.full(n, 11.11)                  # 40 km/h -> az scaled by (22.22 / 11.11)^2 = 4
    raw, ctx = s1_iri.window_tensors(dist, imu6, speed, 0.0)
    assert raw.shape == (400, 6) and raw.dtype == np.float32
    assert ctx.shape == (13,) and ctx.dtype == np.float32
    np.testing.assert_allclose(raw[:, 2], 4.0, rtol=1e-5)
    np.testing.assert_allclose(raw[:, [0, 1, 3, 4, 5]], 1.0)
    # Below 5 m/s the normalisation is capped: (22.22 / 5)^2.
    raw_slow, _ = s1_iri.window_tensors(dist, imu6, np.full(n, 2.0), 0.0)
    np.testing.assert_allclose(raw_slow[:, 2], (22.22 / 5.0) ** 2, rtol=1e-5)


def test_iri_class_bins():
    assert [s1_iri.iri_class(v) for v in (1.2, 3.99, 4.0, 7.9, 8.0, 13.9, 14.0, 20.0)] == \
        ["good", "good", "fair", "fair", "poor", "poor", "very_poor", "very_poor"]


def test_model_binds_inputs_by_name_and_output_is_calibrated():
    pytest.importorskip("tensorflow")
    model = s1_iri.IriModel(repo_path("ML_models/IRI/iri_compliant/iri_background_model.tflite"),
                            repo_path("ML_models/IRI/iri_compliant/mobile_calibration_lut.json"))
    details = {d["index"]: d["name"] for d in model.interpreter.get_input_details()}
    assert "context_stats" in details[model._ctx] and "raw_imu" in details[model._raw]
    # The usage doc binds by position; the real order is the reverse of what it says.
    first_input = model.interpreter.get_input_details()[0]
    assert "context_stats" in first_input["name"]

    rng = np.random.default_rng(1)
    dist = np.linspace(0, 100, 350)
    imu6 = rng.normal(0, 0.5, (350, 6))
    raw, ctx = s1_iri.window_tensors(dist, imu6, np.full(350, 15.0), 0.0)
    iri, iri_raw = model.predict(raw, ctx)
    assert 1.2128 <= iri <= 15.572 and iri_raw > 0
