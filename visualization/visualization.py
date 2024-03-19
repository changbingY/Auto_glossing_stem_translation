import torch
import torch.nn as nn
from transformers import BertTokenizer

import json
import pickle
import numpy as np

language = "uspanteko"
model = "morph"
track = 1


# Load predictions
predictions = pickle.load(open(f"{language.lower()}_{model}_track{track}.prediction.pkl", "rb"))

# Load morpheme tokenizer
morpheme_tokenizer = pickle.load(open(f"{language.lower()}_{model}_track{track}.morpheme_tokenizer.pkl", "rb"))

# Load translation tokenizer (BertTokenizer in this case)
trans_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


all_visualization_items = []
for batch_prediction in predictions:
    _, _, visualization_object = batch_prediction
    attention_weights = visualization_object["attention_weights"]
    translations = visualization_object["translations"]
    decoder_outputs = visualization_object["decoder_outputs"]

    items = []
    batch_size = len(translations["input_ids"])
    assert len(decoder_outputs) == batch_size
    assert len(attention_weights) == batch_size
    for i in range(batch_size):
        item = {}
        translation = trans_tokenizer.convert_ids_to_tokens(translations['input_ids'][i], skip_special_tokens=False)
        translation = [elem for elem in translation if elem not in ['[PAD]']]
        
        decoder_output = decoder_outputs[i] # list with length num_morphemes_per_example

        attention_weight = attention_weights[i]  # [num_morphemes_per_example, max_morpheme_char_length, trans_sequence_length]
        attention_weight = attention_weight[:,:,:len(translation)] # tensor [num_morphemes_per_example, max_morpheme_char_length, actual_trans_sequence_length]

        assert len(decoder_output) == attention_weight.size(0)

        morphemes = []
        num_morphemes_per_example = len(decoder_output)
        for j in range(num_morphemes_per_example):
            morpheme_output = decoder_output[j]
            morpheme_char_length = len(morpheme_output)
            morpheme_output = morpheme_tokenizer.lookup_tokens(morpheme_output)
            if not morpheme_char_length:
                continue
            morpheme = []
            for k, morpheme_chunk in enumerate(morpheme_output):
                morpheme.append((morpheme_chunk, attention_weight[j,k,:].numpy().tolist()))
            morphemes.append(morpheme)
        
        item["translation"] = translation
        item["morphemes"] = morphemes
        items.append(item)
    all_visualization_items.extend(items)


fname = f"{language.lower()}_{model}_track{track}_visualization.json"
with open(fname, 'w') as f:
    json.dump(all_visualization_items, f, ensure_ascii=False, indent=4)
