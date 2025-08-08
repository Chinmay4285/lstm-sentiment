"""
Training script with evaluation and visualization
Trains LSTM model and provides comprehensive performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import SentimentLSTM

def load_sample_data():
    """
    Create sample dataset for demonstration
    In production, replace with your actual dataset loading
    """
    # Sample data for demonstration
    texts = [
        "I love this product! Amazing quality and fast delivery.",
        "Terrible experience, worst purchase ever made.",
        "It's okay, nothing special but not bad either.",
        "Fantastic service and great value for money!",
        "Poor quality, broke after one week of use.",
        "Average product, meets basic expectations.",
        "Outstanding! Exceeded all my expectations completely.",
        "Disappointing results, not worth the price.",
        "Decent quality for the price point offered.",
        "Excellent customer service and quick resolution!"
    ] * 100  # Multiply for larger dataset
    
    labels = [
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "positive", "negative", "neutral", "positive"
    ] * 100
    
    return texts, labels

def plot_training_history(history):
    """
    Visualize training progress with accuracy and loss plots
    
    Args:
        history: Training history from model.fit()
    """
    # Create subplot for accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training and validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax1.set_title('Model Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training and validation loss
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2.set_title('Model Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_texts, test_labels):
    """
    Comprehensive model evaluation with metrics and visualizations
    
    Args:
        model: Trained SentimentLSTM model
        test_texts: List of test text samples
        test_labels: List of true test labels
    """
    # Get predictions for test data
    predictions = model.predict(test_texts)
    
    # Extract predicted classes and true classes
    y_pred = [pred['sentiment'] for pred in predictions]
    y_true = test_labels
    
    # Generate detailed classification report
    print("\n" + "="*50)
    print("DETAILED PERFORMANCE METRICS")
    print("="*50)
    print(classification_report(y_true, y_pred, 
                              target_names=['negative', 'neutral', 'positive']))
    
    # Create and display confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix - Model Performance')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show sample predictions with confidence scores
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    for i, (text, pred) in enumerate(zip(test_texts[:5], predictions[:5])):
        print(f"\nText: {text[:80]}...")
        print(f"Predicted: {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
        print(f"True: {test_labels[i]}")

def main():
    """Main training pipeline"""
    print("🚀 Starting LSTM Sentiment Analysis Training Pipeline")
    print("="*60)
    
    # Load training data (replace with your dataset)
    print("📊 Loading dataset...")
    texts, labels = load_sample_data()
    print(f"Loaded {len(texts)} samples")
    
    # Initialize model with configuration
    print("\n🧠 Initializing LSTM model...")
    model = SentimentLSTM(
        max_words=5000,     # Vocabulary size limit
        max_len=50,         # Maximum sequence length
        embedding_dim=100   # Word embedding dimension
    )
    
    # Split data for training and testing
    split_idx = int(0.8 * len(texts))  # 80% for training
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Testing samples: {len(test_texts)}")
    
    # Train the model
    print("\n🏋️ Training LSTM model...")
    history = model.train(
        train_texts, train_labels,
        epochs=15,           # Number of training epochs
        batch_size=32,       # Batch size for training
        validation_split=0.2 # Use 20% of training data for validation
    )
    
    # Visualize training progress
    print("\n📈 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model performance
    print("\n🔍 Evaluating model performance...")
    evaluate_model(model, test_texts, test_labels)
    
    # Save trained model
    print("\n💾 Saving trained model...")
    model.save_model('sentiment_model')
    print("Model saved as 'sentiment_model_model.h5' and 'sentiment_model_tokenizer.pkl'")
    
    print("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()