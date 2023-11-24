import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict

from torch import Tensor
from torch.autograd import Variable
from itertools import chain
from containers import Batch
from typing import List
from torch.optim import AdamW
from utils import sum_pool_2d
from utils import max_pool_2d
from utils import make_mask_2d
from bert import BERTBasedEncoder
from masked_cross_entropy import *
from bilstm import BiLSTMEncoder
from attention import Attn, LuongAttnDecoderRNN
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR
from morpheme_segmenter import UnsupervisedMorphemeSegmenter

import ipdb

USE_CUDA = torch.cuda.is_available()
PAD_TOKEN = 0
UNK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
MAX_MORPHEME_LENGTH = 15

class MorphemeGlossingModel(LightningModule):
    def __init__(
        self,
        source_alphabet_size: int,
        target_alphabet_size: int,
        target_char_alphabet_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_size: int = 128,
        scheduler_gamma: float = 1.0,
        learn_segmentation: bool = True,
        classify_num_morphemes: bool = False,
    ):
        super().__init__()
        self.source_alphabet_size = source_alphabet_size
        self.target_alphabet_size = target_alphabet_size
        self.target_char_alphabet_size = target_char_alphabet_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.scheduler_gamma = scheduler_gamma
        self.learn_segmentation = learn_segmentation
        self.classify_num_morphemes = classify_num_morphemes

        self.save_hyperparameters()

        self.embeddings = nn.Embedding(
            num_embeddings=self.source_alphabet_size,
            embedding_dim=self.embedding_size,
            padding_idx=0,
        )
        self.encoder = BiLSTMEncoder(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            projection_dim=self.hidden_size,
        )

        self.bert_encoder = BERTBasedEncoder(
            projection_dim = self.hidden_size
        )

        self.decoder = LuongAttnDecoderRNN(
            attn_model="concat",
            hidden_size=self.hidden_size,
            output_size=self.target_char_alphabet_size,
            dropout=self.dropout
        )
        # self.classifier = nn.Linear(self.hidden_size, self.target_alphabet_size)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

        if self.learn_segmentation:
            self.segmenter = UnsupervisedMorphemeSegmenter(self.hidden_size)

        if self.classify_num_morphemes:
            self.num_morpheme_classifier = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 10),
            )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), weight_decay=0.0, lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]

    def encode_sentences(self, sentences: Tensor, sentence_lengths: Tensor) -> Tensor:
        char_embeddings = self.embeddings(sentences)
        char_encodings = self.encoder(char_embeddings, sentence_lengths)
        return char_encodings

    def encode_trans_sentences(self, sentences: List[str], sentence_lengths: Tensor) -> Tensor:
        trans_bert_encodings = self.bert_encoder(sentences, sentence_lengths)
        return trans_bert_encodings  # tensor [batch_size, sequence_length, hidden_size]

    def get_words(self, encodings: Tensor, word_extraction_index: Tensor):
        encodings = encodings.reshape(-1, self.hidden_size)
        num_words, chars_per_word = word_extraction_index.shape
        word_extraction_index_flat = word_extraction_index.flatten()
        word_encodings = torch.index_select(
            encodings, dim=0, index=word_extraction_index_flat
        )
        word_encodings = word_encodings.reshape(
            num_words, chars_per_word, self.hidden_size
        )

        return word_encodings

    def get_num_morphemes(self, word_encodings: Tensor, word_lengths: Tensor):
        assert self.classify_num_morphemes
        word_encodings = max_pool_2d(word_encodings, word_lengths)
        num_morpheme_scores = self.num_morpheme_classifier(word_encodings)
        num_morpheme_predictions = torch.argmax(num_morpheme_scores, dim=-1)
        num_morpheme_predictions = torch.minimum(num_morpheme_predictions, word_lengths)
        num_morpheme_predictions = torch.clip(num_morpheme_predictions, min=1)

        return {"scores": num_morpheme_scores, "predictions": num_morpheme_predictions}

    def get_morphemes(
        self,
        word_encodings: Tensor,
        morpheme_extraction_index: Tensor,
        morpheme_lengths: Tensor,
    ):
        char_encodings = word_encodings.reshape(-1, self.hidden_size)
        num_morphemes, chars_per_morpheme = morpheme_extraction_index.shape
        morpheme_extraction_index = morpheme_extraction_index.flatten()
        morpheme_encodings = torch.index_select(
            char_encodings, dim=0, index=morpheme_extraction_index
        )
        morpheme_encodings = morpheme_encodings.reshape(
            num_morphemes, chars_per_morpheme, self.hidden_size
        )

        # Sum Pool Morphemes
        morpheme_encodings = sum_pool_2d(morpheme_encodings, lengths=morpheme_lengths)

        return morpheme_encodings
    
    def align_translations(self, translation_encodings, num_morphemes_per_word, word_batch_mapping):
        """ Align morphemes with translations for attention computation
        Args:
            translation_encodings: tensor [batch_size, num_trans_chars_per_word, hidden_size]
            num_morphemes_per_word: tensor [num_words]
            word_batch_mapping: list [num_words]
        Return:
            aligned_trans_encodings: tensor [num_morphemes, num_trans_chars_per_word, hidden_size]
        """
        num_morphemes_per_example = defaultdict(int)
        assert len(num_morphemes_per_word) == len(word_batch_mapping)
        for num_morphemes, example_index in zip(num_morphemes_per_word, word_batch_mapping):
            num_morphemes_per_example[example_index] += num_morphemes
        
        aligned_trans_encodings = []
        for example_index, num_morphemes in num_morphemes_per_example.items():
            temp_encodings = translation_encodings[example_index].unsqueeze(0).repeat(num_morphemes, 1, 1)
            aligned_trans_encodings.append(temp_encodings)
        aligned_trans_encodings = torch.cat(aligned_trans_encodings, dim=0)

        return aligned_trans_encodings

    def forward(self, batch: Batch, training: bool = True):
        # ipdb.set_trace()
        char_encodings = self.encode_sentences(
            batch.sentences, batch.sentence_lengths.cpu()
        )
        word_encodings = self.get_words(char_encodings, batch.word_extraction_index)
        trans_bert_encodings = self.encode_trans_sentences(
            batch.trans_sentences_raw, batch.trans_sentence_lengths.cpu()
        )
        # ipdb.set_trace()
        if self.classify_num_morphemes:
            num_morphemes_per_word = self.get_num_morphemes(
                word_encodings, batch.word_lengths
            )
            num_morphemes_per_word_scores = num_morphemes_per_word["scores"]

            if training:
                num_morphemes_per_word = batch.word_target_lengths
            else:
                num_morphemes_per_word = num_morphemes_per_word["predictions"]
        else:
            num_morphemes_per_word_scores = None
            num_morphemes_per_word = batch.word_target_lengths

        if self.learn_segmentation:
            morpheme_encodings, best_path_matrix = self.segmenter(
                word_encodings,
                batch.word_lengths,
                num_morphemes_per_word,
                training=training,
            )

        else:
            assert batch.morpheme_extraction_index is not None
            morpheme_encodings = self.get_morphemes(
                word_encodings, batch.morpheme_extraction_index, batch.morpheme_lengths
            )
            best_path_matrix = None
        # ipdb.set_trace()
        # Align morphemes and translations
        trans_encodings = self.align_translations(trans_bert_encodings, num_morphemes_per_word, batch.word_batch_mapping)
        # ipdb.set_trace()
        # Decoder
        if training:
            num_morphemes = batch.morpheme_char_targets.size(0)
            max_morpheme_char_target_length = batch.morpheme_char_targets.size(1)
            all_decoder_outputs = Variable(torch.zeros(num_morphemes, max_morpheme_char_target_length - 1, self.decoder.output_size))
            decoder_input = batch.morpheme_char_targets[:, 0]
        else:
            num_morphemes = morpheme_encodings.size(0)
            try:
                max_morpheme_char_target_length = batch.morpheme_char_targets.size(1)
            except:
                max_morpheme_char_target_length = MAX_MORPHEME_LENGTH

            #max_morpheme_char_target_length = batch.morpheme_char_targets.size(1)
            all_decoder_outputs = Variable(torch.zeros(num_morphemes, max_morpheme_char_target_length - 1, self.decoder.output_size))
            decoder_input = torch.tensor([START_TOKEN] * num_morphemes).long()

        decoder_hidden = None

        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
        # ipdb.set_trace()
        predicted_char_targets = []
        # Run through decoder one time step at a time
        for t in range(max_morpheme_char_target_length - 1):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(
                input_seq=decoder_input, 
                input_aux=morpheme_encodings,
                last_hidden=decoder_hidden,
                encoder_outputs=trans_encodings
            )

            all_decoder_outputs[:, t, :] = decoder_output
            if training:
                # ipdb.set_trace()
                decoder_input = batch.morpheme_char_targets[:, t + 1] # Next input is current target
            else:
                # ipdb.set_trace()
                # Choose top word from output
                prob, token_idx = decoder_output.data.topk(1)
                predicted_char_targets.append(token_idx)
                
                # Next input is chosen word
                decoder_input = token_idx.clone().detach().squeeze(1)

        return {
            "all_decoder_outputs": all_decoder_outputs,
            "num_morphemes_per_word_scores": num_morphemes_per_word_scores,
            "num_morphemes_per_word": num_morphemes_per_word,
            "best_path_matrix": best_path_matrix,
            "predicted_char_targets": predicted_char_targets,
        }

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        scores = self.forward(batch=batch, training=True)

        # morpheme_classification_loss = self.cross_entropy(
        #     scores["morpheme_scores"], batch.morpheme_targets
        # )
        morpheme_loss = masked_cross_entropy(
            logits=scores["all_decoder_outputs"],
            target=batch.morpheme_char_targets[:, 1:],
            length=batch.morpheme_char_target_lengths - 1
        )

        if self.classify_num_morphemes:
            num_morpheme_loss = self.cross_entropy(
                scores["num_morphemes_per_word_scores"], batch.word_target_lengths
            )
        else:
            num_morpheme_loss = torch.tensor(
                0.0, requires_grad=True, device=self.device
            )
        # ipdb.set_trace()
        loss = (
            morpheme_loss
            + num_morpheme_loss
            - num_morpheme_loss.detach()
        )
        return loss

    @staticmethod
    def get_morpheme_to_word(num_morphemes_per_word: Tensor):
        num_morphemes_per_word_mask = make_mask_2d(num_morphemes_per_word)
        num_morphemes_per_word_mask = torch.logical_not(num_morphemes_per_word_mask)
        num_morphemes_per_word_mask_flat = num_morphemes_per_word_mask.flatten()

        morpheme_to_word = torch.arange(
            num_morphemes_per_word.shape[0], device=num_morphemes_per_word_mask.device
        )
        morpheme_to_word = morpheme_to_word.unsqueeze(1)
        morpheme_to_word = morpheme_to_word.expand(num_morphemes_per_word_mask.shape)
        morpheme_to_word = morpheme_to_word.flatten()
        morpheme_to_word = torch.masked_select(
            morpheme_to_word, mask=num_morphemes_per_word_mask_flat
        )
        morpheme_word_mapping = morpheme_to_word.cpu().tolist()
        return morpheme_word_mapping

    def evaluation_step(self, batch: Batch):
        scores = self.forward(batch=batch, training=False)

        if self.classify_num_morphemes:
            morpheme_word_mapping = self.get_morpheme_to_word(
                scores["num_morphemes_per_word"]
            )
        else:
            morpheme_word_mapping = batch.morpheme_word_mapping
        gold_morpheme_word_mapping = batch.word_target_lengths
        # ipdb.set_trace()
        predicted_indices = torch.argmax(scores["all_decoder_outputs"], dim=-1).cpu().tolist()
        predicted_word_labels = [[] for _ in range(batch.word_lengths.shape[0])]
        for predicted_idx, word_idx in zip(predicted_indices, morpheme_word_mapping):
            pred_idx = [idx for idx in predicted_idx if idx not in SPECIAL_TOKENS]
            predicted_word_labels[word_idx].append(pred_idx)
        # ipdb.set_trace()
        targets = batch.morpheme_char_targets[:, 1:].cpu().tolist()
        target_word_labels = [[] for _ in range(batch.word_lengths.shape[0])]
        # for target, word_idx in zip(targets, gold_morpheme_word_mapping):
        #     target_idx = [idx for idx in target if idx not in SPECIAL_TOKENS]
        #     target_word_labels[word_idx].append(target_idx)
        start_idx = 0
        for word_idx, gold_num_morphemes in enumerate(gold_morpheme_word_mapping):
            target = targets[start_idx: start_idx + gold_num_morphemes]
            target_idx = [[idx for idx in idxes if idx not in SPECIAL_TOKENS] for idxes in target]
            target_word_labels[word_idx].extend(target_idx)
            start_idx += gold_num_morphemes
        
        assert len(target_word_labels) == len(predicted_word_labels)
        # ipdb.set_trace()
        correct = [
            prediction == target
            for prediction, target in zip(predicted_word_labels, target_word_labels)
        ]

        return correct

    def validation_step(self, batch: Batch, batch_idx: int):
        return self.evaluation_step(batch=batch)

    def validation_epoch_end(self, outputs) -> None:
        correct = list(chain.from_iterable(outputs))

        accuracy = np.mean(correct)
        self.log("val_accuracy", 100 * accuracy)

    @staticmethod
    def get_word_segmentations(batch: Batch, best_path_matrix: Tensor):
        word_indices = batch.sentences.flatten()
        word_indices = word_indices[batch.word_extraction_index.flatten()]
        word_indices = word_indices.reshape(batch.word_extraction_index.shape)
        word_indices = word_indices.cpu().tolist()

        max_num_morphemes = best_path_matrix.shape[-1]
        best_path_matrix = best_path_matrix.long()
        best_path_matrix = best_path_matrix.argmax(dim=-1)
        best_path_matrix = best_path_matrix.cpu().tolist()

        word_lengths = batch.word_lengths.cpu().tolist()

        segmentations = []
        for word, word_segmentation_indices, length in zip(
            word_indices, best_path_matrix, word_lengths
        ):
            word_segmentation = [[] for _ in range(max_num_morphemes)]
            word = word[:length]
            word_segmentation_indices = word_segmentation_indices[:length]

            for char_index, segment_index in zip(word, word_segmentation_indices):
                word_segmentation[segment_index].append(char_index)
            word_segmentation = [segment for segment in word_segmentation if segment]
            segmentations.append(word_segmentation)

        sentence_segmentations = [[] for _ in range(batch.sentences.shape[0])]
        for word_segmentation, sentence_idx in zip(
            segmentations, batch.word_batch_mapping
        ):
            sentence_segmentations[sentence_idx].append(word_segmentation)

        return sentence_segmentations

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        scores = self.forward(batch=batch, training=False)

        if self.classify_num_morphemes:
            morpheme_word_mapping = self.get_morpheme_to_word(
                scores["num_morphemes_per_word"]
            )
        else:
            morpheme_word_mapping = batch.morpheme_word_mapping
        #gold_morpheme_word_mapping = batch.word_target_lengths
        # ipdb.set_trace()
        predicted_indices = torch.argmax(scores["all_decoder_outputs"], dim=-1).cpu().tolist()
        predicted_word_labels = [[] for _ in range(batch.word_lengths.shape[0])]
        for predicted_idx, word_idx in zip(predicted_indices, morpheme_word_mapping):
            pred_idx = [idx for idx in predicted_idx if idx not in SPECIAL_TOKENS]
            predicted_word_labels[word_idx].append(pred_idx)
        # ipdb.set_trace()
        #targets = batch.morpheme_char_targets[:, 1:].cpu().tolist()
        #target_word_labels = [[] for _ in range(batch.word_lengths.shape[0])]
        #for target, word_idx in zip(targets, gold_morpheme_word_mapping):
        #    target_idx = [idx for idx in target if idx not in SPECIAL_TOKENS]
        #    target_word_labels[word_idx].append(target_idx)
        #assert len(target_word_labels) == len(predicted_word_labels)

        #return predicted_word_labels, target_word_labels
        #return predicted_word_labels
        predicted_sentence_labels = [[] for _ in range(batch.sentences.shape[0])]
        for word_labels, sentence_idx in zip(
            predicted_word_labels, batch.word_batch_mapping
        ):
            predicted_sentence_labels[sentence_idx].append(word_labels)
        # ipdb.set_trace()
        if scores["best_path_matrix"] is not None:
            learned_segmentation = self.get_word_segmentations(
                batch=batch, best_path_matrix=scores["best_path_matrix"]
            )
        else:
            learned_segmentation = None

        return predicted_sentence_labels, learned_segmentation
