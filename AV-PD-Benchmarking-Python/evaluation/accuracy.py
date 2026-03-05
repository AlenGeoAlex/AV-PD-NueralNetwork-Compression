def evaluate_accuracy(model, dataset):

    metrics = model.val(
    data=dataset,
    verbose=False,
    plots=False,
    save_json=False
)

    return {
        "mAP50": metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr
    }