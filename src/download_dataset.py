import kagglehub

def download_dataset():
    """
    Download dataset from Kaggle.
    """
    path = kagglehub.dataset_download("nhthongl/mind-dataset")
    print("Path to dataset files:", path)
