"""NPU configuration regressions without loading accelerator dependencies."""

import pytest

from deployment.npu.config import recommended_deploy_params


@pytest.mark.parametrize(
    "tp,context,memory", [(1, 32768, 0.70), (2, 8192, 0.70), (4, 32768, 0.70)]
)
def test_default_profiles(tp, context, memory):
    assert recommended_deploy_params(tp, {}) == {
        "context_length": context,
        "mem_fraction_static": memory,
        "max_running_requests": tp,
    }


def test_explicit_settings_override_defaults():
    env = {
        "CONTEXT_LENGTH": "12345",
        "MEM_FRACTION": "0.6",
        "MAX_RUNNING_REQUESTS": "3",
    }
    assert recommended_deploy_params(2, env) == {
        "context_length": 12345,
        "mem_fraction_static": 0.6,
        "max_running_requests": 3,
    }
    assert env["MEM_FRACTION"] == "0.6"


@pytest.mark.parametrize(
    "key,value",
    [
        ("CONTEXT_LENGTH", "0"),
        ("CONTEXT_LENGTH", "bad"),
        ("MEM_FRACTION", "nan"),
        ("MEM_FRACTION", "inf"),
        ("MEM_FRACTION", "0"),
        ("MEM_FRACTION", "1.1"),
        ("MAX_RUNNING_REQUESTS", "-1"),
    ],
)
def test_invalid_settings_fail_before_startup(key, value):
    with pytest.raises(ValueError):
        recommended_deploy_params(2, {key: value})


@pytest.mark.parametrize("tp", [0, -1, True, 1.5])
def test_invalid_tp_size(tp):
    with pytest.raises(ValueError):
        recommended_deploy_params(tp, {})
