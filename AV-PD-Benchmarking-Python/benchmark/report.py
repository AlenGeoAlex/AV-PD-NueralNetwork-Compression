import csv
import os

def save_results_csv(results, output_path="results/benchmark_results.csv"):

    os.makedirs("results", exist_ok=True)

    fieldnames = [
        "model",
        "mAP50",
        "mAP50_95",
        "latency",
        "fps",
        "parameters",
        "size_mb"
    ]

    with open(output_path, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for model_name, metrics in results.items():

            row = {
                "model": model_name,
                "mAP50": metrics["mAP50"],
                "mAP50_95": metrics["mAP50_95"],
                "latency": metrics["latency"],
                "fps": metrics["fps"],
                "parameters": metrics["parameters"],
                "size_mb": metrics["size_mb"]
            }

            writer.writerow(row)

    print(f"Results saved to {output_path}")