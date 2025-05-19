import torch

class GPUTimer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.diff = 0.
        self.avg = 0.
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.started = False 

    def tic(self):
        self.start.record()
        self.started = True  

    def toc(self, printed_text="", average=False):
        if not self.started:
            raise RuntimeError("Error: `tic()` must be called before `toc()`")

        self.end.record()
        self.end.synchronize() 

        t_msec = self.start.elapsed_time(self.end)
        self.diff = t_msec
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls

        if average:
            return self.avg
        else:
            return self.diff
