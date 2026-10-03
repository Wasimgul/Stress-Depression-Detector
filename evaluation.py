import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import clean_text

def evaluate_model():
    model = tf.keras.models.load_model('models/stress_model.h5')
    
    sample_texts = [
        "I am feeling extremely stressed out and anxious today.",
        "Having a peaceful and wonderful walk in the park."
    ]
    
    cleaned_texts = [clean_text(text) for text in sample_texts]
    
    max_words = 5000
    max_len = 100
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)
    
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    predictions = model.predict(padded)
    
    for text, pred in zip(sample_texts, predictions):
        status = "Stress/Depression Detected" if pred[0] > 0.5 else "Normal/Positive"
        print(f"Text: {text}")
        print(f"Prediction Score: {pred[0]:.4f} -> {status}\n")

if __name__ == "__main__":
    evaluate_model()





