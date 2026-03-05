import os

def get_model_size(model_path):
    """
    Returns model size in MB
    """

    size_bytes = os.path.getsize(model_path)

    size_mb = size_bytes / (1024 * 1024)

    return size_mb