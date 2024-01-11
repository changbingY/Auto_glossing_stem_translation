from collections import namedtuple

GlossingFileData = namedtuple(
    "GlossingFileData", field_names=["sources", "targets", "morphemes","translations"]
)
Batch = namedtuple(
    "Batch",
    [
        "sentences",
        "sentence_lengths",
        "word_lengths",
        "word_extraction_index",
        "word_batch_mapping",
        "word_targets",
        "word_target_lengths",
        "trans_sentences_raw",
        "trans_sentences",
        "trans_sentence_lengths",
        "trans_word_lengths",
        "trans_word_extraction_index",
        "trans_word_batch_mapping",
        "morpheme_extraction_index",
        "morpheme_lengths",
        "morpheme_word_mapping",
        "morpheme_targets",
    ],
)
Hyperparameters = namedtuple(
    "Hyperparameters",
    field_names=[
        "batch_size",
        "num_layers",
        "hidden_size",
        "dropout",
        "scheduler_gamma",
    ],
)
