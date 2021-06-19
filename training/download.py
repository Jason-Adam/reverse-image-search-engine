import os

import gdown

DATASET_URL = "https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp"

if __name__ == "__main__":
    dirs = os.listdir()

    if "caltech101" not in dirs:
        gdown.download(DATASET_URL)
