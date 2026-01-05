import torch
import numpy as np
import os

def get_last_line(inputfile):
    filesize = os.path.getsize(inputfile)
    blocksize = 200
    dat_file = open(inputfile, 'rb')
    last_line = ""
    if filesize > blocksize:
        maxseekpoint = (filesize // blocksize)
        dat_file.seek((maxseekpoint - 1) * blocksize)
    elif filesize:
        dat_file.seek(0, 0)
    lines = dat_file.readlines()
    if lines:
        last_line = lines[-1].strip()
    dat_file.close()
    return last_line


class CUDATimer:
    def __init__(self, device=None):
        self.device = device or torch.cuda.current_device()
        self.start_event = None
        self.end_event = None
        self.elapsed_time_ms = 0
    def __enter__(self):
        torch.cuda.synchronize(self.device)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self
    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
    def get_time_ms(self):
        """time(ms)"""
        return self.elapsed_time_ms
    def get_time_s(self):
        """tims(s)"""
        return self.elapsed_time_ms / 1000.0


def print_scientific(t):
    for key, value in t.items():
        if isinstance(value, (float)):
            print(f' {key}: {value:.2e}', end=' ')
        else:
            print(f' {key}: {value}', end=' ')
    print('')