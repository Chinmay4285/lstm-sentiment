from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
max_len = 100

def load_sentiment_model():
    """Load the trained LSTM model and tokenizer"""
    global model, tokenizer
    try:
        # Load the trained model
        model = load_model('models/sentiment_model.h5')
        
        # Load the tokenizer
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        print("✅ Model and tokenizer loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize and remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [w for w in word_tokens if w not in stop_words]
        text = ' '.join(filtered_text)
    except:
        # If NLTK data not available, just return cleaned text
        pass
    
    return text

def predict_sentiment(text):
    """Predict sentiment for given text"""
    global model, tokenizer, max_len
    
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Convert text to sequence
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        
        # Pad sequence
        padded_sequences = pad_sequences(sequences, maxlen=max_len)
        
        # Make prediction
        prediction = model.predict(padded_sequences, verbose=0)[0]
        
        # Map predictions to labels
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        
        # Create probabilities dictionary
        probabilities = {
            'negative': float(prediction[0]),
            'neutral': float(prediction[1]),
            'positive': float(prediction[2])
        }
        
        return {
            'sentiment': sentiment_labels[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities,
            'status': 'success'
        }
        
    except Exception as e:
        return {'error': str(e), 'status': 'error'}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment for single text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided', 'status': 'error'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided', 'status': 'error'}), 400
        
        result = predict_sentiment(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze sentiment for multiple texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided', 'status': 'error'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be a list', 'status': 'error'}), 400
        
        results = []
        for text in texts:
            if text.strip():
                result = predict_sentiment(text)
                results.append({
                    'text': text,
                    'analysis': result
                })
            else:
                results.append({
                    'text': text,
                    'analysis': {'error': 'Empty text', 'status': 'error'}
                })
        
        return jsonify({
            'results': results,
            'total_analyzed': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    tokenizer_status = "loaded" if tokenizer is not None else "not_loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'tokenizer_status': tokenizer_status,
        'service': 'lstm_sentiment_analysis'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

if __name__ == '__main__':
    print("🚀 Starting LSTM Sentiment Analysis Web Application...")
    
    # Download NLTK data if needed
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data ready")
    except:
        print("⚠️ NLTK data download failed, continuing without stopwords")
    
    # Load the model
    if load_sentiment_model():
        print("🌐 Starting Flask server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please ensure you have:")
        print("   1. Trained your model using: python train.py")
        print("   2. Model files exist in the 'models/' directory")
        print("   3. Files: sentiment_model.h5 and tokenizer.pickle")