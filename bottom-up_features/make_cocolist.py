import os
from unicodedata import name
import pickle


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, "jpg")
    img_ids = set()
    for img in images:
        img_id = int(img.split("/")[-1].split(".")[0].split("_")[-1])
        img_ids.add(img_id)
    return img_ids


if __name__ == "__main__":
    train_imageids = load_imageid("/opt/ml/input/data/train2014")
    valid_imageids = load_imageid("/opt/ml/input/data/val2014")

    with open("./train_imageids.pkl", "wb") as f:
        pickle.dump(train_imageids, f)

    with open("./valid_imageids.pkl", "wb") as f:
        pickle.dump(valid_imageids, f)
