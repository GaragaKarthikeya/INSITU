"""Llama 3's frequency rescaling: the regimes, and the config path into it."""

import numpy as np

from kernel.config import ModelConfig
from kernel.ops.rope import RoPE, RopeScaling

# Llama-3.2-1B's values.
S = RopeScaling(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0,
                original_max_position=8192)
THETA = 500000.0
D = 64


def _inv_freq(d=D, theta=THETA):
    return 1.0 / (theta ** (np.arange(0, d // 2, dtype=np.float64) * 2.0 / d))


def check_short_wavelengths_are_untouched():
    """High-frequency channels carry local position and must not be stretched."""
    f = _inv_freq()
    out = S.apply(f)
    wavelen = 2.0 * np.pi / f
    short = wavelen < S.original_max_position / S.high_freq_factor
    assert short.any(), "test vector has no short-wavelength channels"
    assert np.array_equal(out[short], f[short])


def check_long_wavelengths_are_divided_by_factor():
    f = _inv_freq()
    out = S.apply(f)
    wavelen = 2.0 * np.pi / f
    long = wavelen > S.original_max_position / S.low_freq_factor
    assert long.any(), "test vector has no long-wavelength channels"
    assert np.allclose(out[long], f[long] / S.factor)


def check_the_middle_band_is_monotone_and_bounded():
    """The interpolation exists to avoid a discontinuity; check it delivers one.

    Every scaled frequency lies between `f / factor` and `f`, and the scaling
    ratio moves in one direction across the band -- a sign error in `smooth`
    passes the two regime tests above and fails here.
    """
    f = _inv_freq()
    out = S.apply(f)
    assert np.all(out <= f * (1 + 1e-12))
    assert np.all(out >= f / S.factor * (1 - 1e-12))
    ratio = out / f
    assert np.all(np.diff(ratio) <= 1e-12), "ratio must be non-increasing in wavelength"


def check_scaling_changes_the_table_and_nothing_else():
    """The claim the module docstring makes: same shapes, same trace, new angles."""
    from kernel.trace import Trace

    plain = RoPE.build(D, 64, THETA)
    scaled = RoPE.build(D, 64, THETA, scaling=S)
    assert plain.cos.shape == scaled.cos.shape and plain.frac == scaled.frac
    assert not np.array_equal(plain.cos, scaled.cos)

    x = np.arange(D, dtype=np.int64) * 7
    pos = np.array([5])
    ta, tb = Trace(), Trace()
    plain.apply(x[None, :], pos, 12, ta, "u")
    scaled.apply(x[None, :], pos, 12, tb, "u")
    assert [(w.op, w.unit, w.m, w.k, w.n) for w in ta.records] == \
           [(w.op, w.unit, w.m, w.k, w.n) for w in tb.records]


def check_position_zero_is_unaffected_by_scaling():
    """Angle is position * frequency: at position 0 the scaling cannot show."""
    plain = RoPE.build(D, 8, THETA)
    scaled = RoPE.build(D, 8, THETA, scaling=S)
    assert np.array_equal(plain.cos[0], scaled.cos[0])
    assert np.array_equal(plain.sin[0], scaled.sin[0])


def check_rejects_an_inverted_frequency_band():
    """`high <= low` divides by zero in the interpolation; it must not build."""
    for bad in (dict(low_freq_factor=4.0, high_freq_factor=1.0),
                dict(low_freq_factor=2.0, high_freq_factor=2.0)):
        try:
            RopeScaling(factor=32.0, original_max_position=8192, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


class _Llama32Config:
    hidden_size = 2048
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 64
    max_position_embeddings = 131072
    rope_parameters = {
        "rope_type": "llama3", "rope_theta": THETA, "factor": 32.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }


def check_from_hf_carries_llama3_scaling():
    m = ModelConfig.from_hf(_Llama32Config)
    assert m.rope_theta == THETA
    assert m.rope_scaling == S


def check_from_hf_still_refuses_scalings_that_are_not_modelled():
    class _Yarn(_Llama32Config):
        rope_parameters = dict(_Llama32Config.rope_parameters, rope_type="yarn")

    try:
        ModelConfig.from_hf(_Yarn)
    except NotImplementedError as e:
        assert "yarn" in str(e)
        return
    raise AssertionError("yarn scaling was accepted")


def check_default_rope_type_yields_no_scaling():
    class _Plain(_Llama32Config):
        rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}

    assert ModelConfig.from_hf(_Plain).rope_scaling is None
