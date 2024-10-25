# content of conftest.py


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "npu: indicate whether test is supported on npu")
