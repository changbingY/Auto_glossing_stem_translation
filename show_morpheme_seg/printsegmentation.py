import pickle

language = "statimcets"
model = "morph"
track = 1

# Load predictions
predictions = pickle.load(
    open(f"{language.lower()}_{model}_track{track}.prediction.pkl", "rb"))

# Load source tokenizer
source_tokenizer = pickle.load(
    open(f"{language.lower()}_{model}_track{track}.source_tokenizer.pkl",
         "rb"))

for prediction in predictions:
    _, segmentations = prediction
    for sentence in segmentations:
        for word in sentence:
            morphs = []
            for morph in word:
                morphs.append(source_tokenizer.lookup_tokens(morph))
            print(morphs)
