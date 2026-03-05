from models.loader import load_model
from evaluation.latency import measure_latency
from evaluation.params import count_parameters
from evaluation.model_size import get_model_size
from evaluation.accuracy import evaluate_accuracy


def run_benchmark(model_path, dataset, image):

    model = load_model(model_path)

    accuracy = evaluate_accuracy(model, dataset)

    latency = measure_latency(model, image)

    params = count_parameters(model)

    size = get_model_size(model_path)

    fps = 1 / latency

    return {
        "mAP50": accuracy["mAP50"],
        "mAP50_95": accuracy["mAP50_95"],
        "latency": latency,
        "fps": fps,
        "parameters": params,
        "size_mb": size
    }