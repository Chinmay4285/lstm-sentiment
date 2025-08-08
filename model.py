"""
LSTM Sentiment Analysis Model
Complete implementation with bidirectional LSTM layers and preprocessing
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split

# Download required NLTK data once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SentimentLSTM:
    """
    Complete LSTM sentiment analysis pipeline
    Handles preprocessing, model building, training, and prediction
    """
    
    def __init__(self, max_words=10000, max_len=100, embedding_dim=128):
        """
        Initialize LSTM sentiment analyzer
        
        Args:
            max_words (int): Maximum vocabulary size for tokenizer
            max_len (int): Maximum sequence length for padding
            embedding_dim (int): Dimension of word embedding vectors
        """
        self.max_words = max_words      # Vocabulary limit for memory efficiency
        self.max_len = max_len          # Standardizes input sequence lengths
        self.embedding_dim = embedding_dim  # Dense vector representation size
        
        # Initialize tokenizer for text-to-sequence conversion
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        
        # NLTK components for text preprocessing
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
        
        # Model will be built during training
        self.model = None
    
    def clean_text(self, text):
        """
        Comprehensive text preprocessing pipeline
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned and processed text
        """
        # Convert to lowercase for case-insensitive processing
        text = text.lower()
        
        # Remove URLs (http, https, www patterns)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions (@username) and hashtags (#hashtag)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Keep only alphabetical characters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize text into individual words
        words = nltk.word_tokenize(text)
        
        # Remove stopwords and short words, apply lemmatization
        words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Rejoin words into cleaned sentence
        return ' '.join(words)
    
    def prepare_data(self, texts, labels=None, fit_tokenizer=True):
        """
        Convert text data into model-ready numerical format
        
        Args:
            texts (list): List of text strings
            labels (list): List of sentiment labels (optional)
            fit_tokenizer (bool): Whether to fit tokenizer on this data
            
        Returns:
            tuple: (X_sequences, y_categorical) or just X_sequences
        """
        # Clean all texts using preprocessing pipeline
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        if fit_tokenizer:
            # Build vocabulary from training data
            self.tokenizer.fit_on_texts(cleaned_texts)
        
        # Convert texts to integer sequences based on vocabulary
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # Pad sequences to uniform length (truncate/pad as needed)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        if labels is not None:
            # Convert string labels to numerical format
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            y = np.array([label_map[label.lower()] for label in labels])
            
            # Convert to one-hot encoding for categorical crossentropy
            y = to_categorical(y, num_classes=3)
            return X, y
        
        return X
    
    def build_model(self):
        """
        Construct bidirectional LSTM architecture
        
        Returns:
            keras.Model: Compiled LSTM model
        """
        model = Sequential([
            # Word embedding layer - converts integer sequences to dense vectors
            Embedding(
                input_dim=self.max_words,      # Vocabulary size
                output_dim=self.embedding_dim, # Dense vector dimension
                input_length=self.max_len,     # Fixed sequence length
                mask_zero=True                 # Ignore padded zeros
            ),
            
            # Dropout prevents overfitting by randomly setting inputs to 0
            Dropout(0.3),
            
            # Bidirectional LSTM - processes sequence forward and backward
            Bidirectional(LSTM(
                128,                    # Hidden units in each direction
                return_sequences=True,  # Return full sequence (not just last output)
                dropout=0.3,           # Dropout within LSTM layer
                recurrent_dropout=0.2  # Dropout on recurrent connections
            )),
            
            # Second bidirectional layer with fewer units
            Bidirectional(LSTM(
                64,                     # Fewer units for hierarchical learning
                dropout=0.3,
                recurrent_dropout=0.2
            )),
            
            # Dense layer with ReLU activation for non-linearity
            Dense(64, activation='relu'),
            Dropout(0.5),  # Higher dropout before output layer
            
            # Output layer: 3 neurons for 3 classes (neg, neutral, pos)
            Dense(3, activation='softmax')  # Softmax for probability distribution
        ])
        
        # Compile model with optimizer and loss function
        model.compile(
            optimizer='adam',                    # Adaptive learning rate optimizer
            loss='categorical_crossentropy',     # Standard loss for multi-class
            metrics=['accuracy']                 # Track accuracy during training
        )
        
        self.model = model
        return model
    
    def train(self, texts, labels, validation_split=0.2, epochs=10, batch_size=32):
        """
        Train the LSTM model on provided data
        
        Args:
            texts (list): Training text samples
            labels (list): Training sentiment labels
            validation_split (float): Fraction of data for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History: Training history object
        """
        # Prepare training data
        X, y = self.prepare_data(texts, labels, fit_tokenizer=True)
        
        # Build model architecture
        if self.model is None:
            self.build_model()
        
        # Train model with validation split
        history = self.model.fit(
            X, y,
            validation_split=validation_split,  # Use portion of data for validation
            epochs=epochs,                      # Number of training iterations
            batch_size=batch_size,              # Samples per gradient update
            verbose=1                           # Show training progress
        )
        
        return history
    
    def predict(self, texts):
        """
        Make sentiment predictions on new texts
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            list: Predicted sentiments with confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")
        
        # Prepare data (don't fit tokenizer - use existing vocabulary)
        X = self.prepare_data(texts, fit_tokenizer=False)
        
        # Get prediction probabilities
        predictions = self.model.predict(X, verbose=0)
        
        # Convert probabilities to readable format
        sentiment_labels = ['negative', 'neutral', 'positive']
        results = []
        
        for pred in predictions:
            # Get index of highest probability
            sentiment_idx = np.argmax(pred)
            sentiment = sentiment_labels[sentiment_idx]
            confidence = float(pred[sentiment_idx])
            
            results.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    'negative': float(pred[0]),
                    'neutral': float(pred[1]),
                    'positive': float(pred[2])
                }
            })
        
        return results
    
    def save_model(self, filepath):
        """Save trained model and tokenizer"""
        if self.model:
            # Save model architecture and weights
            self.model.save(f"{filepath}_model.h5")
            
            # Save tokenizer for consistent preprocessing
            import pickle
            with open(f"{filepath}_tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
    
    def load_model(self, filepath):
        """Load pre-trained model and tokenizer"""
        # Load model
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        
        # Load tokenizer
        import pickle
        with open(f"{filepath}_tokenizer.pkl", 'rb') as f:
            self.tokenizer = pickle.load(f)