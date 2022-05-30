import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pickle
from transformers import AutoTokenizer


def create_input_files(tokenizer_name, json_path, captions_per_image, output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset. Since bottom up features only available for coco, we use only coco
    :param json_path: path of ko coco JSON file with splits and captions
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Read Karpathy JSON
    with open(json_path, "r") as j:
        data = json.load(j)

    with open(os.path.join(output_folder, "train36_imgid2idx.pkl"), "rb") as j:
        train_data = pickle.load(j)

    with open(os.path.join(output_folder, "val36_imgid2idx.pkl"), "rb") as j:
        val_data = pickle.load(j)

    # Read image paths and captions for each image
    train_image_captions = []
    val_image_captions = []
    test_image_captions = []
    train_image_det = []
    val_image_det = []
    test_image_det = []
    word_freq = Counter()

    for idx, row in enumerate(data):
        if "val" in row["file_path"]:  # ko_coco에는 split key가 없기 때문에 만들어줌.
            if idx >= 20252:
                row["split"] = "test"
            else:
                row["split"] = "valid"
        else:
            row["split"] = "train"

        captions = []
        for cap in row["caption_ko"]:
            # Update word frequency
            if len(tokenizer.tokenize(cap)) <= max_len:
                captions.append(cap)

        if len(captions) == 0:
            continue

        image_id = row["id"]

        if row["split"] == "train":
            train_image_det.append(("t", train_data[image_id]))
            train_image_captions.append(captions)

        elif row["split"] == "valid":
            val_image_det.append(("v", val_data[image_id]))
            val_image_captions.append(captions)
        elif row["split"] == "test":
            test_image_det.append(("v", val_data[image_id]))
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_det) == len(train_image_captions)
    assert len(val_image_det) == len(val_image_captions)
    assert len(test_image_det) == len(test_image_captions)

    # Create a base/root name for all output files
    base_filename = str(captions_per_image) + "_cap_per_img"

    for impaths, imcaps, split in [
        (train_image_det, train_image_captions, "TRAIN"),
        (val_image_det, val_image_captions, "VAL"),
        (test_image_det, test_image_captions, "TEST"),
    ]:
        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(impaths)):
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [
                    choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))
                ]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            for j, c in enumerate(captions):
                # Encode captions
                enc_c = tokenizer.encode(
                    c, max_length=max_len, padding="max_length", truncation=True,
                )

                enc_captions.append(enc_c)
                caplens.append(len(enc_c) - enc_c.count(0))

        # Save encoded captions and their lengths to JSON files
        with open(
            os.path.join(output_folder, split + "_CAPTIONS_" + base_filename + ".json"), "w"
        ) as j:
            json.dump(enc_captions, j)

        with open(
            os.path.join(output_folder, split + "_CAPLENS_" + base_filename + ".json"), "w"
        ) as j:
            json.dump(caplens, j)

    # Save bottom up features indexing to JSON files
    with open(
        os.path.join(output_folder, "TRAIN" + "_GENOME_DETS_" + base_filename + ".json"), "w"
    ) as j:
        json.dump(train_image_det, j)

    with open(
        os.path.join(output_folder, "VAL" + "_GENOME_DETS_" + base_filename + ".json"), "w"
    ) as j:
        json.dump(val_image_det, j)

    with open(
        os.path.join(output_folder, "TEST" + "_GENOME_DETS_" + base_filename + ".json"), "w"
    ) as j:
        json.dump(test_image_det, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def save_checkpoint(
    data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, bleu4, is_best
):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param decoder: decoder model
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "bleu-4": bleu4,
        "decoder": decoder,
        "decoder_optimizer": decoder_optimizer,
    }
    filename = "checkpoint_" + data_name + ".pth.tar"
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, "BEST_" + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
