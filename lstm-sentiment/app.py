"""
Flask web application for real-time sentiment analysis
Provides RESTful API and web interface for LSTM model
"""

from flask import Flask, render_template, request, jsonify
import json
from model import SentimentLSTM

# Initialize Flask application
app = Flask(__name__)

# Global model instance (loaded once when app starts)
model = None

def load_model():
    """
    Load pre-trained LSTM model on application startup
    Global model prevents reloading on each request
    """
    global model
    try:
        model = SentimentLSTM()
        model.load_model('sentiment_model')
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/')
def home():
    """
    Serve main web interface
    
    Returns:
        Rendered HTML template for sentiment analysis interface
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    API endpoint for sentiment analysis
    Accepts JSON with 'text' field and returns sentiment prediction
    
    Returns:
        JSON response with sentiment, confidence, and probabilities
    """
    try:
        # Extract text from JSON request
        data = request.get_json()
        text = data.get('text', '').strip()
        
        # Validate input
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) > 1000:  # Prevent extremely long inputs
            return jsonify({'error': 'Text too long (max 1000 characters)'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Make prediction using LSTM model
        predictions = model.predict([text])
        result = predictions[0]  # Get first (and only) prediction
        
        # Format response with detailed information
        response = {
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': round(result['confidence'], 3),
            'probabilities': {
                'negative': round(result['probabilities']['negative'], 3),
                'neutral': round(result['probabilities']['neutral'], 3),
                'positive': round(result['probabilities']['positive'], 3)
            },
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        # Log error and return error response
        print(f"Error in sentiment analysis: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """
    API endpoint for analyzing multiple texts at once
    Accepts JSON with 'texts' array
    
    Returns:
        JSON array with sentiment analysis for each text
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        # Validate input
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'No texts provided or invalid format'}), 400
        
        if len(texts) > 50:  # Limit batch size
            return jsonify({'error': 'Too many texts (max 50)'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Analyze all texts at once (efficient batch processing)
        predictions = model.predict(texts)
        
        # Format results
        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            results.append({
                'index': i,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': pred['sentiment'],
                'confidence': round(pred['confidence'], 3),
                'probabilities': {
                    'negative': round(pred['probabilities']['negative'], 3),
                    'neutral': round(pred['probabilities']['neutral'], 3),
                    'positive': round(pred['probabilities']['positive'], 3)
                }
            })
        
        return jsonify({
            'results': results,
            'total_analyzed': len(texts),
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error in batch analysis: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring
    
    Returns:
        JSON with service status and model availability
    """
    model_status = 'loaded' if model is not None else 'not_loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'service': 'lstm_sentiment_analysis'
    })

if __name__ == '__main__':
    print("🌐 Starting LSTM Sentiment Analysis Web Service...")
    
    # Load model before starting server
    if load_model():
        print("🚀 Starting Flask server...")
        # Run in debug mode for development
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Cannot start server - model loading failed!")
        print("Make sure you've trained the model by running: python train.py")