"""Sample NVML process/device memory without synchronizing the model's GPU work."""

import os
import threading
import time

from common import write_json

# Reference capacity, not a test pass/fail threshold.
BUDGET_BYTES = 80_000_000_000


class MemoryMonitor:
    def __init__(self, path, interval=0.05):
        self.path = path
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        try:
            import pynvml as _nvml_mod

            self.nvml = _nvml_mod
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None

    def sample(self):
        if self.nvml is None:
            return
        info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        used = sum(
            p.usedGpuMemory
            for p in processes
            if p.pid == os.getpid()
            and isinstance(p.usedGpuMemory, int)
            and 0 <= p.usedGpuMemory <= info.total
        )
        self.samples.append(
            dict(
                elapsed_seconds=time.monotonic() - self.started,
                process_bytes=used,
                device_bytes=info.used,
            )
        )

    def __enter__(self):
        if self.nvml is None:
            self.started = time.monotonic()
            self.total = 0
            self.thread = None
            return self
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.handle = (
            pynvml.nvmlDeviceGetHandleByUUID(visible)
            if visible.startswith(("GPU-", "MIG-"))
            else pynvml.nvmlDeviceGetHandleByIndex(int(visible))
        )
        self.started = time.monotonic()
        self.total = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
        self.sample()

        def collect():
            while not self.stop.wait(self.interval):
                try:
                    self.sample()
                except Exception as exc:
                    self.errors.append(repr(exc))
                    break

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        if getattr(self, "thread", None) is not None:
            self.thread.join(5)
        try:
            self.sample()
        except Exception as error:
            self.errors.append(repr(error))
        if self.nvml is None:
            write_json(
                self.path,
                dict(
                    interval_seconds=self.interval,
                    pid=os.getpid(),
                    device_total_bytes=0,
                    note="nvml unavailable (NPU)",
                ),
            )
            return
        self.nvml.nvmlShutdown()
        process_peak = max(s["process_bytes"] for s in self.samples)
        device_peak = max(s["device_bytes"] for s in self.samples)
        write_json(
            self.path,
            dict(
                interval_seconds=self.interval,
                pid=os.getpid(),
                device_total_bytes=self.total,
                process_peak_bytes=process_peak,
                device_peak_bytes=device_peak,
                budget_bytes=BUDGET_BYTES,
                measurement_valid=0 < process_peak <= device_peak <= self.total
                and not self.errors,
                within_budget=0 < process_peak <= device_peak <= BUDGET_BYTES
                and not self.errors,
                errors=self.errors,
                samples=self.samples,
            ),
        )
