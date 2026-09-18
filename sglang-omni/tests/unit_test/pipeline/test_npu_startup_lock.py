"""Device-key regressions; no accelerator runtime is required."""

from pathlib import Path

import pytest

from sglang_omni.utils.npu_startup import get_npu_startup_lock_path


def test_pinned_npu_ranks_use_distinct_keys():
    first = get_npu_startup_lock_path(0, env={"ASCEND_RT_VISIBLE_DEVICES": "5"})
    second = get_npu_startup_lock_path(0, env={"ASCEND_RT_VISIBLE_DEVICES": "6"})
    assert first.name == "sglang_omni_npu_5_startup.lock"
    assert second.name == "sglang_omni_npu_6_startup.lock"
    assert first != second


def test_npu_key_is_stable_across_visibility_layouts():
    pinned = get_npu_startup_lock_path(0, env={"ASCEND_RT_VISIBLE_DEVICES": "5"})
    unpinned = get_npu_startup_lock_path(
        2, env={"ASCEND_RT_VISIBLE_DEVICES": "0,1,5,6"}
    )
    assert pinned == unpinned


def test_cuda_visibility_is_not_used_for_npu():
    path = get_npu_startup_lock_path(
        0,
        env={"ASCEND_RT_VISIBLE_DEVICES": "9", "CUDA_VISIBLE_DEVICES": "GPU-other"},
        base_dir="/tmp/example",
    )
    assert path == Path("/tmp/example/sglang_omni_npu_9_startup.lock")


def test_without_npu_visibility_uses_device_index():
    assert get_npu_startup_lock_path(3, env={}).name == "sglang_omni_npu_3_startup.lock"


@pytest.mark.parametrize("visible", ["1,,2", "-1", "1,bad", "1,2,"])
def test_invalid_npu_visibility_is_rejected(visible):
    with pytest.raises(ValueError):
        get_npu_startup_lock_path(0, env={"ASCEND_RT_VISIBLE_DEVICES": visible})


def test_invalid_local_index_is_not_treated_as_a_physical_id():
    with pytest.raises(ValueError):
        get_npu_startup_lock_path(1, env={"ASCEND_RT_VISIBLE_DEVICES": "5"})
