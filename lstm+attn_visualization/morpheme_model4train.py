import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from torch import Tensor
from itertools import chain
from containers import Batch
from torch.optim import AdamW
from utils import sum_pool_2d
from utils import max_pool_2d
from utils import make_mask_2d
from bilstm import BiLSTMEncoder
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR
from morpheme_segmenter import UnsupervisedMorphemeSegmenter
from attention import Attn
from masked_cross_entropy import *
from collections import defaultdict

#.cuda()
#.cpu()

class MorphemeGlossingModel(LightningModule):
    def __init__(
        self,
        source_alphabet_size: int,
        target_alphabet_size: int,
        translation_alphabet_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_size: int = 128,
        scheduler_gamma: float = 1.0,
        learn_segmentation: bool = True,
        classify_num_morphemes: bool = False,
    ):
        super().__init__()
        self.attn = Attn('general', hidden_size)
        self.source_alphabet_size = source_alphabet_size
        self.target_alphabet_size = target_alphabet_size
        self.translation_alphabet_size = translation_alphabet_size
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

        self.translation_embeddings = nn.Embedding(
            num_embeddings=self.translation_alphabet_size,
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

        self.translation_encoder = BiLSTMEncoder(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            projection_dim=self.hidden_size,
        )

        self.classifier = nn.Sequential(
             nn.Linear(self.hidden_size * 2, self.hidden_size), 
             nn.GELU(),
             nn.Linear(self.hidden_size, self.target_alphabet_size)
        )
        #self.classifier = nn.Linear(self.hidden_size*2, self.target_alphabet_size)
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

    def encode_trans_sentences(self, sentences: Tensor, sentence_lengths: Tensor) -> Tensor:
        char_embeddings = self.translation_embeddings(sentences)
        char_encodings= self.translation_encoder(char_embeddings, sentence_lengths)
        return char_encodings

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
    
    def align_morphemes_with_translations(self, morpheme_encodings, translation_encodings, num_morphemes_per_word, word_batch_mapping, trans_word_batch_mapping):
        """ Align morphemes with translations for attention computation
        Args:
            morpheme_encodings: tensor [num_morphemes, hidden_size]
            translation_encodings: tensor [num_trans_words, num_trans_chars_per_word, hidden_size]
            num_morphemes_per_word: tensor [num_words]
            word_batch_mapping: list [num_words]
            trans_word_batch_mapping: list [num_trans_words]
        Return:
            morpheme_encodings_per_batch: dict, key batch_id, value tensor [num_morphemes_per_batch, hidden_size]
            trans_encodings_per_batch: dict, key batch_id, value tensor [num_trans_words_per_batch, hidden_size]
        """
        num_morphemes_per_batch = defaultdict(int)
        assert len(num_morphemes_per_word) == len(word_batch_mapping)
        for num_morphemes, batch_index in zip(num_morphemes_per_word, word_batch_mapping):
            num_morphemes_per_batch[batch_index] += num_morphemes
        print('num_morphemes_per_batch')
        print(num_morphemes_per_batch)
        morpheme_encodings_per_batch = {}
        morpheme_index = 0
        for batch_index, num_morphemes in num_morphemes_per_batch.items():
            morpheme_encodings_per_batch[batch_index] = morpheme_encodings[morpheme_index:morpheme_index+ num_morphemes]
            morpheme_index += num_morphemes
    
        trans_encodings_per_batch = defaultdict(list)
        translation_encodings = torch.mean(translation_encodings, dim=1)  # [num_trans_words, hidden_size]
        assert translation_encodings.size()[0] == len(trans_word_batch_mapping)
        for i in range(len(trans_word_batch_mapping)):
            batch_index = trans_word_batch_mapping[i]
            trans_encodings_per_batch[batch_index].append(translation_encodings[i])
        for batch_index in trans_encodings_per_batch.keys():
            trans_encodings_per_batch[batch_index] = torch.stack(trans_encodings_per_batch[batch_index], dim=0)
    
        return morpheme_encodings_per_batch, trans_encodings_per_batch

    def forward(self, batch: Batch, training: bool = True):
        char_encodings = self.encode_sentences(
            batch.sentences, batch.sentence_lengths.cpu()
        )
        
        word_encodings = self.get_words(char_encodings, batch.word_extraction_index)
        trans_char_encodings = self.encode_trans_sentences(
            batch.trans_sentences, batch.trans_sentence_lengths.cpu()
        )
        translation_word_encodings = self.get_words(trans_char_encodings, batch.trans_word_extraction_index)
        
        #word_encodings = torch.cat((word_encodings,translation_word_encodings))
        #translation_word_encodings = pad_sequence([word_encodings,translation_word_encodings])
        #print('trans_word_encodings')
        #print(translation_word_encodings.size())
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
            #trans_morpheme_encodings, trans_best_path_matrix = self.segmenter(
             #   translation_word_encodings,
              #  batch.trans_word_lengths,
              #  num_morphemes_per_word,
               # training=training,
            #)
        else:
            assert batch.morpheme_extraction_index is not None
            morpheme_encodings = self.get_morphemes(
                word_encodings, batch.morpheme_extraction_index, batch.morpheme_lengths
            )
            best_path_matrix = None
            #trans_best_path_matrix = None
            #trans_morpheme_encodings = self.get_morphemes(
            #translation_word_encodings, batch.morpheme_extraction_index, batch.morpheme_lengths)
            #print('trans_morpheme_encoding')
            #print(trans_morpheme_encodings.size())
        
       # translation_word_encodings = torch.mean(translation_word_encodings, dim=(0,1)).unsqueeze(0)
        translation_attention_encodings = []
        #print('original_morpheme_encodings')
        #print(morpheme_encodings.size())
        #morpheme_encodings = torch.mean(morpheme_encodings,dim=1)
        #print('mean_morpheme_encodings')
        #print(morpheme_encodings.size())
        #align_morphemes_with_translations(morpheme_encodings, translation_encodings, num_morphemes_per_word, word_batch_mapping, trans_word_batch_mapping)
        morpheme_encodings_per_batch, trans_encodings_per_batch = self.align_morphemes_with_translations(morpheme_encodings, translation_word_encodings, num_morphemes_per_word, batch.word_batch_mapping, batch.trans_word_batch_mapping)
        #for each_morpheme_encoding in morpheme_encodings:
            #print('each_morpheme_encoding')
            #print(each_morpheme_encoding.size())
            #attn_score = self.attn(each_morpheme_encoding,translation_word_encodings)
            #each_attention_trans = torch.sum(attn_score.cuda() * translation_word_encodings.cuda(),dim=0)
            #each_attention_trans = torch.sum(attn_score * translation_word_encodings,dim=0)
            
        translation_attention_encodings = []
        for batch_index in morpheme_encodings_per_batch.keys():
            translation_attention_encodings_this_batch = []
            morpheme_encodings_this_batch = morpheme_encodings_per_batch[batch_index]
            trans_encodings_this_batch = trans_encodings_per_batch[batch_index]
            if not morpheme_encodings_this_batch.size()[0]:
                continue
            for each_morpheme_encoding in morpheme_encodings_this_batch:
                attn_score = self.attn(each_morpheme_encoding, trans_encodings_this_batch)
                each_attention_trans = torch.sum(attn_score * trans_encodings_this_batch, dim=0)
                translation_attention_encodings_this_batch.append(each_attention_trans)
                #print('each_attention_trans')
                #print(each_attention_trans.size())
            #print('translation_attention_encodings_this_batch')
            #print(len(translation_attention_encodings_this_batch))
            translation_attention_encodings_this_batch = torch.stack(translation_attention_encodings_this_batch)
            #print('translation_attention_encodings_this_batch_after_stack')
            #print(translation_attention_encodings_this_batch.size())
            
            translation_attention_encodings.append(translation_attention_encodings_this_batch)
            #print('translation_attention_encodings_length')
            #print(len(translation_attention_encodings))
        translation_attention_encodings = torch.cat(translation_attention_encodings, dim=0)
        #translation_attention_encodings = torch.stack(translation_attention_encodings)
        #print('morpheme_encodings')
        #print(morpheme_encodings.size())
        #print('translation_attention_encodings')
        #print(translation_attention_encodings.size())
        #combined_encodings = torch.cat(
        #    (morpheme_encodings.cpu(),
        #    translation_attention_encodings.cpu()),dim=1
        #)

        combined_encodings = torch.cat(
            (morpheme_encodings.cuda(),
            translation_attention_encodings.cuda()),dim=1
        )


        #combined_encodings = torch.cat(
         #   (morpheme_encodings,
         #   translation_attention_encodings),dim=1
        #)
        #combined_encodings = torch.cat((morpheme_encodings,translation_word_encodings))
        
        morpheme_scores = self.classifier(combined_encodings)

        return {
            "num_morphemes_per_word_scores": num_morphemes_per_word_scores,
            "num_morphemes_per_word": num_morphemes_per_word,
            "morpheme_scores": morpheme_scores,
            "best_path_matrix": best_path_matrix,
        }

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        scores = self.forward(batch=batch, training=True)

        morpheme_classification_loss = self.cross_entropy(
            scores["morpheme_scores"], batch.morpheme_targets
        )
        if self.classify_num_morphemes:
            num_morpheme_loss = self.cross_entropy(
                scores["num_morphemes_per_word_scores"], batch.word_target_lengths
            )
        else:
            num_morpheme_loss = torch.tensor(
                0.0, requires_grad=True, device=self.device
            )

        loss = (
            morpheme_classification_loss
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

        predicted_indices = (
            torch.argmax(scores["morpheme_scores"], dim=-1).cpu().tolist()
        )

        predicted_word_labels = [[] for _ in range(batch.word_lengths.shape[0])]
        for predicted_idx, word_idx in zip(predicted_indices, morpheme_word_mapping):
            predicted_word_labels[word_idx].append(predicted_idx)

        targets = batch.word_targets.cpu().tolist()
        targets = [[idx for idx in target if idx != 0] for target in targets]
        assert len(targets) == len(predicted_word_labels)

        correct = [
            prediction == target
            for prediction, target in zip(predicted_word_labels, targets)
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

        predicted_indices = (
            torch.argmax(scores["morpheme_scores"], dim=-1).cpu().tolist()
        )

        predicted_word_labels = [[] for _ in range(batch.word_lengths.shape[0])]
        for predicted_idx, word_idx in zip(predicted_indices, morpheme_word_mapping):
            predicted_word_labels[word_idx].append(predicted_idx)

        predicted_sentence_labels = [[] for _ in range(batch.sentences.shape[0])]
        for word_labels, sentence_idx in zip(
            predicted_word_labels, batch.word_batch_mapping
        ):
            predicted_sentence_labels[sentence_idx].append(word_labels)

        if scores["best_path_matrix"] is not None:
            learned_segmentation = self.get_word_segmentations(
                batch=batch, best_path_matrix=scores["best_path_matrix"]
            )
        else:
            learned_segmentation = None

        return predicted_sentence_labels, learned_segmentation
