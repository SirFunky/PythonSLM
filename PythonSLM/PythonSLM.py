
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import nltk
from nltk.tokenize import word_tokenize

# Sample text corpus (You can replace this with your own dataset)
text_corpus = """Deep learning is amazing. Neural networks are powerful. 
                 Machine learning is changing the world. AI is the future."""

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_corpus])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
words = word_tokenize(text_corpus.lower())  # Convert text to tokens

for i in range(1, len(words)):
    n_gram_sequence = words[:i+1]
    encoded_sequence = tokenizer.texts_to_sequences([" ".join(n_gram_sequence)])[0]
    input_sequences.append(encoded_sequence)

# Pad sequences to ensure equal length
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Split into features (X) and labels (y)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the LSTM model
model = Sequential([
    Embedding(total_words, 10, input_length=max_sequence_length-1),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(50, activation='relu'),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to generate text
def generate_text(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate new text
print(generate_text("Machine learning", next_words=5))