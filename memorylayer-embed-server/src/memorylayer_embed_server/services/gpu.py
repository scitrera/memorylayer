"""GPU status monitoring service."""

from logging import Logger

from scitrera_app_framework import Variables, get_logger


class GPUStatusMonitor:
    """
    Monitors GPU status and memory usage.

    Provides health check information for the /health/gpu endpoint.
    """

    def __init__(self, v: Variables = None):
        self.logger: Logger = get_logger(v, name=self.__class__.__name__)
        self._torch_available = None

    def _check_torch(self) -> bool:
        """Check if torch is available."""
        if self._torch_available is None:
            try:
                import torch  # noqa: F401

                self._torch_available = True
            except ImportError:
                self._torch_available = False
        return self._torch_available

    def get_gpu_status(self) -> dict:
        """
        Get current GPU status information.

        Returns dict with:
        - available: bool
        - device_count: int
        - devices: list of device info dicts
        """
        if not self._check_torch():
            return {
                "available": False,
                "device_count": 0,
                "devices": [],
                "error": "PyTorch not installed",
            }

        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "available": False,
                    "device_count": 0,
                    "devices": [],
                }

            device_count = torch.cuda.device_count()
            devices = []

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                mem_info = torch.cuda.mem_get_info(i)
                free_mem, total_mem = mem_info

                devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_mb": round(total_mem / (1024 * 1024)),
                        "free_memory_mb": round(free_mem / (1024 * 1024)),
                        "used_memory_mb": round((total_mem - free_mem) / (1024 * 1024)),
                        "utilization_pct": round((1 - free_mem / total_mem) * 100, 1) if total_mem > 0 else 0,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )

            return {
                "available": True,
                "device_count": device_count,
                "devices": devices,
            }

        except Exception as e:
            self.logger.warning("Failed to get GPU status: %s", e)
            return {
                "available": False,
                "device_count": 0,
                "devices": [],
                "error": str(e),
            }
