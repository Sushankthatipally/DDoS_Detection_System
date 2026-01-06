"""
Deep Neural Network Model for DDoS Attack Detection
This model is designed for binary classification (Attack vs Normal)
"""

import tensorflow as tf
from tensorflow import keras

def get_model(input_shape):
    """
    Creates a Deep Neural Network for DDoS detection
    
    Args:
        input_shape: Tuple representing the input shape (num_features,)
    
    Returns:
        Compiled Keras model
    """
    model = keras.models.Sequential([
        # Input layer + First hidden layer
        keras.layers.Dense(64, activation='relu', input_shape=input_shape, name='dense_1'),
        keras.layers.Dropout(0.3, name='dropout_1'),  # Prevent overfitting
        
        # Second hidden layer
        keras.layers.Dense(32, activation='relu', name='dense_2'),
        keras.layers.Dropout(0.2, name='dropout_2'),
        
        # Third hidden layer (for complex patterns)
        keras.layers.Dense(16, activation='relu', name='dense_3'),
        
        # Output layer (Binary classification: 0=Normal, 1=Attack)
        keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_model_summary(input_shape):
    """
    Prints the model architecture
    """
    model = get_model(input_shape)
    print("\n" + "="*60)
    print("🧠 DDoS DETECTION MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60 + "\n")
    return model


# Test the model
if __name__ == "__main__":
    # Test with sample input shape
    from data_utils import get_input_shape
    
    print("🔍 Testing model creation...")
    input_shape = get_input_shape()
    print(f"✅ Input shape detected: {input_shape}")
    
    model = get_model_summary(input_shape)
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {model.count_params():,}")
