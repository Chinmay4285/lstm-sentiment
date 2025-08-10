import os
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print("🚀 Starting LSTM Sentiment Analysis Training Pipeline")

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK data downloaded")
except:
    print("⚠️ NLTK download failed, continuing without stopwords")

def create_sample_dataset(n_samples=2000):
    """Create a sample dataset for training"""
    print(f"📊 Creating sample dataset with {n_samples} samples...")
    
    # Positive examples
    positive_texts = [
        "I love this product! It's amazing and exceeded my expectations.",
        "Fantastic quality! Highly recommend to everyone.",
        "Outstanding service and excellent product quality.",
        "This is the best purchase I've made in years!",
        "Incredible value for money. Five stars!",
        "Perfect! Exactly what I was looking for.",
        "Wonderful experience from start to finish.",
        "Top-notch quality and fast delivery.",
        "Brilliant product! Will definitely buy again.",
        "Awesome! Better than I imagined.",
        "Great product, great price, great service!",
        "Excellent quality and fantastic customer support.",
        "Amazing! This product is a game changer.",
        "Perfect quality and fast shipping.",
        "Love it! Highly satisfied with this purchase.",
        "Outstanding! Exceeded all my expectations.",
        "Fantastic! Worth every penny.",
        "Great value and excellent performance.",
        "Superb quality! Couldn't be happier.",
        "Perfect! Exactly as described."
    ]
    
    # Negative examples
    negative_texts = [
        "Terrible quality! Complete waste of money.",
        "Worst product ever. Don't buy this junk.",
        "Poor quality and terrible customer service.",
        "Horrible experience! I want my money back.",
        "Cheaply made and breaks easily.",
        "Not worth the money. Very disappointed.",
        "Bad quality and slow delivery.",
        "Awful! Nothing like the description.",
        "Poor craftsmanship and overpriced.",
        "Disappointing quality for the price.",
        "Terrible! Broke after one use.",
        "Poor quality control and bad design.",
        "Horrible! Worst purchase ever made.",
        "Cheap materials and poor construction.",
        "Not recommended. Save your money.",
        "Terrible experience from start to finish.",
        "Poor quality and not as advertised.",
        "Disappointing! Not worth the price.",
        "Bad product with terrible support.",
        "Waste of money! Very poor quality."
    ]
    
    # Neutral examples
    neutral_texts = [
        "It's okay, nothing special but decent.",
        "Average product for the price.",
        "Not bad, but not great either.",
        "It does what it's supposed to do.",
        "Acceptable quality for the price point.",
        "It's fine, meets basic expectations.",
        "Decent product, nothing extraordinary.",
        "Average quality and standard service.",
        "It works as expected, nothing more.",
        "Okay product, could be better.",
        "Standard quality for this price range.",
        "It's alright, serves its purpose.",
        "Fair quality and reasonable price.",
        "Adequate but not impressive.",
        "It's decent enough for occasional use.",
        "Satisfactory but not outstanding.",
        "It's functional but nothing special.",
        "Reasonable quality for the cost.",
        "It works fine, no complaints.",
        "Average product, does the job."
    ]
    
    # Generate samples
    texts = []
    labels = []
    
    samples_per_class = n_samples // 3
    
    # Positive samples
    for i in range(samples_per_class):
        text = positive_texts[i % len(positive_texts)]
        # Add some variation
        if i > len(positive_texts):
            text = text.replace("product", "item") if "product" in text else text
            text = text.replace("great", "excellent") if "great" in text else text
        texts.append(text)
        labels.append(2)  # positive
    
    # Negative samples
    for i in range(samples_per_class):
        text = negative_texts[i % len(negative_texts)]
        if i > len(negative_texts):
            text = text.replace("terrible", "awful") if "terrible" in text else text
            text = text.replace("bad", "poor") if "bad" in text else text
        texts.append(text)
        labels.append(0)  # negative
    
    # Neutral samples
    for i in range(samples_per_class):
        text = neutral_texts[i % len(neutral_texts)]
        if i > len(neutral_texts):
            text = text.replace("okay", "alright") if "okay" in text else text
            text = text.replace("decent", "acceptable") if "decent" in text else text
        texts.append(text)
        labels.append(1)  # neutral
    
    print(f"✅ Created {len(texts)} samples: {samples_per_class} each for positive, negative, neutral")
    return texts, labels

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords (optional)
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [w for w in word_tokens if w not in stop_words and len(w) > 2]
        text = ' '.join(filtered_text)
    except:
        pass
    
    return text

def prepare_data(texts, labels, max_words=5000, max_len=100):
    """Prepare data for training"""
    print("🔧 Preprocessing data...")
    
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Tokenize
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Convert labels to categorical
    y = tf.keras.utils.to_categorical(labels, num_classes=3)
    
    print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} sequence length")
    print(f"📊 Vocabulary size: {len(tokenizer.word_index)}")
    
    return X, y, tokenizer

def create_model(vocab_size, embedding_dim=128, max_len=100):
    """Create bidirectional LSTM model"""
    print("🏗️ Building Bidirectional LSTM model...")
    
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=15):
    """Train the model"""
    print("🏋️ Training model...")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("📊 Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    class_names = ['Negative', 'Neutral', 'Positive']
    report = classification_report(y_true_classes, y_pred_classes, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    print("\n📈 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Accuracy
    accuracy = report['accuracy']
    print(f"\n🎯 Overall Accuracy: {accuracy:.3f}")
    
    return report

def save_model_and_tokenizer(model, tokenizer, model_dir='models'):
    """Save trained model and tokenizer"""
    print("💾 Saving model and tokenizer...")
    
    # Create models directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'sentiment_model.h5')
    model.save(model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Tokenizer saved to: {tokenizer_path}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🧠 LSTM SENTIMENT ANALYSIS TRAINING")
    print("=" * 60)
    
    # Parameters
    MAX_WORDS = 5000
    MAX_LEN = 100
    EMBEDDING_DIM = 128
    EPOCHS = 15
    N_SAMPLES = 2000
    
    # Create or load dataset
    texts, labels = create_sample_dataset(n_samples=N_SAMPLES)
    
    # Prepare data
    X, y, tokenizer = prepare_data(texts, labels, max_words=MAX_WORDS, max_len=MAX_LEN)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"📊 Data splits:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    # Create model
    model = create_model(vocab_size=MAX_WORDS, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN)
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS)
    
    # Evaluate model
    report = evaluate_model(model, X_test, y_test)
    
    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer)
    
    print("\n🎉 Training completed successfully!")
    print("🌐 You can now run the web application: python app.py")

if __name__ == "__main__":
    main()