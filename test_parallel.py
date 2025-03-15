import multiprocessing
import cupy
from cupy import cuda

class Worker(multiprocessing.Process):
    def __init__(self, gpu_id):
        super().__init__()
        self.gpu_id = gpu_id
    def run(self):
        with cuda.Device(self.gpu_id):
            cupy.asarray(12345)
            print(f"Using GPU {self.gpu_id}")

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    gpu_ids = range(cuda.runtime.getDeviceCount())
    workers = [Worker(gpu_id) for gpu_id in gpu_ids]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()