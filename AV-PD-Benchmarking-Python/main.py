from benchmark.runner import run_benchmark
from benchmark.report import save_results_csv

dataset = "coco128.yaml"
image = "bus.jpg"

models = {
    "teacher_yolo11x": "weights/yolo11x.pt",
    "student_yolo11s": "weights/yolo11s.pt",
}

results = {}

for name, path in models.items():

    print(f"\nRunning benchmark for {name}")

    results[name] = run_benchmark(path, dataset, image)

save_results_csv(results)