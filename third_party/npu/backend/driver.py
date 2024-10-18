from triton.backends.driver import NPUDriverBase
from triton.backends.compiler import NPUTarget

# ------------------------
# Utils
# ------------------------

class NPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    @staticmethod
    def get_device_properties(device):
        # This is just dummy for now. We will need to implement driver.c.
        return {
            "max_shared_mem": 0,
            "multiprocessor_count": 0,
            "sm_clock_rate": 0,
            "mem_clock_rate": 0,
            "mem_bus_width": 0,
        }

    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        # This is just dummy for now. We will need to implement driver.c.
        return (None, kernel_asm, 0, 0)

# ------------------------
# Launcher
# ------------------------

def make_launcher(constants, signature, ids):
    pass

class NPULauncher(object):
    def __init__(self, src, metadata):
        # TODO:
        self.launch = lambda *args, **kwargs: None
    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)

class NPUDriver(NPUDriverBase):

    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        super().__init__()

    def get_current_target(self):
        # Capability are zeros for NPU.
        return NPUTarget("npu", 0)

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        import torch
        return torch.cuda