"""系统状态 API — 提供 GPU / 内存 / 系统监控数据"""
import os
import platform
import shutil
import subprocess
import time
from fastapi import APIRouter

router = APIRouter()

_BOOT_TIME = time.time()


def _gpu_info():
    """通过 nvidia-smi 获取 GPU 名称、利用率和显存"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        parts = [p.strip() for p in out.strip().split(", ")]
        return {
            "gpu_name": parts[0],
            "gpu_util": int(parts[1]),
            "mem_used_mb": int(parts[2]),
            "mem_total_mb": int(parts[3]),
            "gpu_temp": int(parts[4]),
        }
    except Exception:
        return {
            "gpu_name": "N/A",
            "gpu_util": 0,
            "mem_used_mb": 0,
            "mem_total_mb": 0,
            "gpu_temp": 0,
        }


def _cpu_mem_info():
    """读取 /proc 获取 CPU 核心数和系统内存"""
    cpu_count = os.cpu_count() or 0
    mem_total = 0
    mem_avail = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_total = int(line.split()[1]) // 1024  # MB
                elif line.startswith("MemAvailable"):
                    mem_avail = int(line.split()[1]) // 1024
    except Exception:
        pass
    return {
        "cpu_count": cpu_count,
        "sys_mem_total_mb": mem_total,
        "sys_mem_used_mb": mem_total - mem_avail,
    }


def _disk_info():
    total, used, free = shutil.disk_usage("/")
    return {
        "disk_total_gb": round(total / (1 << 30), 1),
        "disk_used_gb": round(used / (1 << 30), 1),
    }


def _uptime_str():
    elapsed = int(time.time() - _BOOT_TIME)
    h, m = divmod(elapsed // 60, 60)
    return f"{h}h {m}m"


@router.get("/status")
def system_status():
    gpu = _gpu_info()
    cpu_mem = _cpu_mem_info()
    disk = _disk_info()
    return {
        "gpu_name": gpu["gpu_name"],
        "gpu_util": gpu["gpu_util"],
        "gpu_temp": gpu["gpu_temp"],
        "mem_used_mb": gpu["mem_used_mb"],
        "mem_total_mb": gpu["mem_total_mb"],
        "cpu_count": cpu_mem["cpu_count"],
        "sys_mem_total_mb": cpu_mem["sys_mem_total_mb"],
        "sys_mem_used_mb": cpu_mem["sys_mem_used_mb"],
        "disk_total_gb": disk["disk_total_gb"],
        "disk_used_gb": disk["disk_used_gb"],
        "python_version": platform.python_version(),
        "uptime": _uptime_str(),
    }
