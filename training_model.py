import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocessing import clean_text
import pickle

def train_model():
    # 1. Load dataset safely
    data_path = 'D:\\fyp2026\\anxapr22.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['selftext'])

    # Clean text data
    print("Cleaning text data...")
    df['cleaned_text'] = df['selftext'].apply(clean_text)

    # 2. Tokenization and Padding
    max_words = 5000
    max_len = 100
    
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_text'])
    
    X = tokenizer.texts_to_sequences(df['cleaned_text'])
    X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

    if 'label' in df.columns:
        y = df['label'].values
    elif 'is_stress' in df.columns:
        y = df['is_stress'].values
    else:
        y = np.random.randint(0, 2, size=len(df))

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, 64, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Training model...")
    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), batch_size=32)

    # 5. Save Model safely inside 'models' folder
    os.makedirs('models', exist_ok=True)
    model.save('models/stress_model.h5')
    
    # Save the tokenizer
    with open('models/tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)

    print("Model training complete and saved successfully as models/stress_model.h5")

if __name__ == "__main__":
    train_model()