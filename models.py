from multiprocessing.sharedctypes import Value
from random import random
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import gensim
from numpy.random import choice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(
            nn.Linear(features_dim, attention_dim)
        )  # linear layer to transform encoded image
        self.decoder_att = weight_norm(
            nn.Linear(decoder_dim, attention_dim)
        )  # linear layer to transform decoder's output
        self.full_att = weight_norm(
            nn.Linear(attention_dim, 1)
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(
            2
        )  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, features_dim)

        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        pretrained_emb,
        decoder_dim,
        bert_model,
        vocab,
        vocab_size,
        features_dim=2048,
        dropout=0.5,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.pretrained_emb = pretrained_emb
        self.bert_model = bert_model
        self.attention = Attention(features_dim, decoder_dim, attention_dim)  # attention network

        if self.pretrained_emb == True:
            model = gensim.models.fasttext.load_facebook_model("./ko.bin")
            wv = model.wv
            vectors = wv.vectors
            word_embeddings = torch.zeros((vocab_size, 200))
            for index, (key, value) in enumerate(vocab.items()):
                if wv.key_to_index.get(key) == None:
                    continue
                else:
                    vec = vectors[wv.key_to_index.get(key)]
                    word_embeddings[value] = torch.FloatTensor(vec)
            self.embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        elif self.bert_model is not None:
            self.embedding = bert_model
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.epsilon = 1
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(
            embed_dim + features_dim + decoder_dim, decoder_dim, bias=True
        )  # top down attention LSTMCell
        self.language_model = nn.LSTMCell(
            features_dim + decoder_dim, decoder_dim, bias=True
        )  # language model LSTMCell
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.fc = weight_norm(
            nn.Linear(decoder_dim, vocab_size)
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        if self.pretrained_emb == True or self.bert_model is not None:
            pass
        else:
            self.embedding.weight.data.uniform_(-0.1, 0.1)
            print("emb layer set uniform")
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Flatten image
        image_features_mean = image_features.mean(1).to(
            device
        )  # (batch_size, num_pixels, encoder_dim)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        if self.bert_model is None:
            embeddings = self.embedding(
                encoded_captions
            )  # (batch_size, max_caption_length, embed_dim)
        else:
            embeddings = self.embedding(
                encoded_captions
            ).last_hidden_state  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions1 = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model
        for t in range(max(decode_lengths)):
            # decode_lengths?????? batch??? ???????????? sequence??? ????????? ?????????????????? ????????????.
            # [26,25,25,22,16,,,,,]
            # 0?????? for?????? ???????????? ????????? Token ??? ????????? ?????? ????????? batch_size_t??? ??????.
            # ???????????? ?????? 18????????? ????????? ????????? sequence??? t??? 18??? ???????????? ????????? ????????? ????????? ??????, 19??? ??? ????????? ?????? x
            batch_size_t = sum([l > t for l in decode_lengths])

            coin_flip = choice(
                [True, False], 1, p=[self.epsilon, 1 - self.epsilon]
            )  # for Scheduled sampling
            if coin_flip == True or t == 0:
                word_embedding = embeddings[:batch_size_t, t, :]
            else:
                word_embedding = pred_embedding[:batch_size_t, :]

            h1, c1 = self.top_down_attention(
                torch.cat(
                    [
                        h2[:batch_size_t],  # (batch, 1024)
                        image_features_mean[:batch_size_t],  # (batch, 2048)
                        word_embedding,  # (batch, 1024)
                    ],
                    dim=1,
                ),  # concat = (batch, 4096)
                (h1[:batch_size_t], c1[:batch_size_t]),
            )

            attention_weighted_encoding = self.attention(
                image_features[:batch_size_t], h1[:batch_size_t]
            )
            preds1 = self.fc1(self.dropout(h1))
            h2, c2 = self.language_model(
                torch.cat([attention_weighted_encoding[:batch_size_t], h1[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]),
            )
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            predictions1[:batch_size_t, t, :] = preds1

            # for Scheduled sampling
            scores = self.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            next_word = scores.argmax(dim=1)
            if self.bert_model is None:
                pred_embedding = self.embedding(next_word)
            else:
                pred_embedding = self.embedding(next_word.unsqueeze(1)).last_hidden_state

        return predictions, predictions1, encoded_captions, decode_lengths, sort_ind
