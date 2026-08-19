from functools import cache
import os


@cache
def get_compute_devices() -> list[str]:
    """
    List all compute devices available.
    Relies on CUDA_VISIBLE_DEVICES for CUDA-compatible devices.
    Uses torch format (eg: "cuda:5")
    """
    import subprocess, re, platform

    devices = []

    # CUDA
    try:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        devices += [f"cuda:{i}" for i in cuda_visible_devices.split(",")]
    except:
        pass

    # MPS
    if platform.system() == "Darwin":
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)

        if re.search(r"Apple.*GPU", out):
            devices.append("mps")

    # CPU
    devices.append("cpu")

    return devices
