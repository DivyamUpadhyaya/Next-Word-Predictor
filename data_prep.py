import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(filepath="sample_text.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read().lower().split("\n")
    return data

def prepare_sequences(corpus, vocab_size=5000, max_len=5):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(corpus)

    sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram = token_list[:i+1]
            sequences.append(n_gram)

    max_sequence_len = max([len(seq) for seq in sequences])
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding="pre"))

    X = sequences[:, :-1]
    y = sequences[:, -1]
    return X, y, tokenizer, max_sequence_len
