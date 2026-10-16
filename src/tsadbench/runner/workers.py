"""Number of worker processes of the per-channel parallel stages (SMED, distance sweeps)."""
import os

CGROUP = "/sys/fs/cgroup"


def cgroup_free(root: str = CGROUP) -> int | None:
    """Bytes the memory limit of the cgroup leaves free, the page cache counted as free (the
    kernel reclaims it at the limit, and the download and conversion of the dataset leave
    gigabytes of it charged to the container); None without a limit."""
    try:
        with open(f"{root}/memory.max") as f:
            limit = f.read().strip()
        if limit == "max":
            return None
        with open(f"{root}/memory.current") as f:
            used = int(f.read())
        with open(f"{root}/memory.stat") as f:
            cache = next((int(line.split()[1]) for line in f if line.startswith("file ")), 0)
        return int(limit) - used + cache
    except (OSError, ValueError):
        return None


def available_memory() -> int | None:
    """Bytes of memory available: ``MemAvailable``, capped by the cgroup limit inside a
    container; None where neither can be read (Windows, macOS)."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    free = cgroup_free()
    if free is not None:
        avail = free if avail is None else min(avail, free)
    return avail


def worker_count(n_tasks: int, cap: int | None = None, gb_per_worker: float = 1.5) -> int:
    """Processes for ``n_tasks`` independent channels: one per core, at most ``cap`` and
    ``TSADBENCH_WORKERS`` when set, and few enough to leave ``gb_per_worker`` GB of the
    available memory to each (a full-set channel takes up to about 1 GB)."""
    n = min(n_tasks, os.cpu_count() or 1)
    if cap:
        n = min(n, cap)
    if os.environ.get("TSADBENCH_WORKERS"):
        n = min(n, int(os.environ["TSADBENCH_WORKERS"]))
    mem = available_memory()
    if mem is not None:
        n = min(n, int(mem // (gb_per_worker * 2**30)))
    return max(1, n)
