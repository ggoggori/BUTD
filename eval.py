import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# Parameters
data_folder = "final_dataset"  # folder with data files saved by create_input_files.py
data_name = "5_cap_per_img"  # base name shared by data files
checkpoint_file = "/opt/ml/input/BUTD/checkpoint_5_cap_per_img.pth.tar"  # model checkpoint
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file, map_location=device)
decoder = checkpoint["decoder"]
decoder = decoder.to(device)
decoder.eval()

# Load word map (word2ix)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
vocab_size = tokenizer.vocab_size


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, "TEST"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
    )

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image_features, caps, caplens, allcaps) in enumerate(
        tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))
    ):

        k = beam_size

        # Move to GPU device, if available
        image_features = image_features.to(device)
        image_features_mean = image_features.mean(1).to(
            device
        )  # (batch_size, num_pixels, encoder_dim)
        image_features_mean = image_features_mean.expand(k, 2048)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[tokenizer.cls_token_id]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
        h2, c2 = decoder.init_hidden_state(k)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            h1, c1 = decoder.top_down_attention(
                torch.cat([h2, image_features_mean, embeddings], dim=1), (h1, c1)
            )  # (batch_size_t, decoder_dim)
            attention_weighted_encoding = decoder.attention(image_features, h1)
            h2, c2 = decoder.language_model(
                torch.cat([attention_weighted_encoding, h1], dim=1), (h2, c2)
            )

            scores = decoder.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode="trunc")  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
            )  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != tokenizer.sep_token_id
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        except:
            seq = [2, 22460, 3]
        # References
        img_caps = allcaps[0].tolist()
        # img_caps = tokenizer.batch_decode(img_caps, skip_special_tokens=True)

        img_captions = list(
            map(
                lambda c: [
                    w
                    for w in c
                    if w
                    not in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
                ],
                img_caps,
            )
        )  # remove <start> and pads

        references.append(img_captions)

        # Hypotheses
        # hypothesis = tokenizer.decode(seq, skip_special_tokens=True)
        hypothesis = [
            w
            for w in seq
            if w not in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        ]
        hypotheses.append(hypothesis)
        assert len(references) == len(hypotheses)

    # Calculate scores
    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4


if __name__ == "__main__":
    beam_size = 1
    metrics = evaluate(beam_size)
    print(metrics)
