import time
import torch

def measure_latency(model, image):

    warmup = 20
    runs = 100

    for _ in range(warmup):
        model(image, verbose=False)

    start = time.time()

    for _ in range(runs):
        model(image, verbose=False)

    end = time.time()

    latency = (end - start) / runs

    return latency