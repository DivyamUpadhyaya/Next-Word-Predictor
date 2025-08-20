import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model + tokenizer
model = load_model("next_word_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer, max_sequence_len = pickle.load(f)

def predict_next_word(seed_text, top_n=3):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")

    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-top_n:][::-1]

    predictions = [(word, predicted_probs[idx]) for word, idx in tokenizer.word_index.items() if idx in top_indices]
    return predictions

# Example
if __name__ == "__main__":
    while True:
        seed = input("Enter a sentence: ")
        if seed.lower() == "exit":
            break
        print("Predictions:", predict_next_word(seed))
