import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from data_prep import load_data, prepare_sequences

# Load data
corpus = load_data("sample_text.txt")

# Prepare sequences
X, y, tokenizer, max_sequence_len = prepare_sequences(corpus)

# One-hot encode labels
y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)

# Model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 50, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index)+1, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
model.fit(X, y, epochs=200, verbose=1)

# Save model + tokenizer
model.save("next_word_model.h5")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump((tokenizer, max_sequence_len), f)

print("✅ Training complete! Model and tokenizer saved.")
