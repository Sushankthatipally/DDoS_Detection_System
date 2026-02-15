"""
Deep Neural Network Model for DDoS Attack Detection
This model is designed for binary classification (Attack vs Normal)
Now with improved regularization to prevent 100% accuracy overfitting
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

def get_model(input_shape):
    """
    Creates a Deep Neural Network for DDoS detection with proper regularization
    
    Args:
        input_shape: Tuple representing the input shape (num_features,)
    
    Returns:
        Compiled Keras model
    """
    model = keras.models.Sequential([
        # Input layer + First hidden layer with L2 regularization
        keras.layers.Dense(
            128, 
            activation='relu', 
            input_shape=input_shape, 
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_1'
        ),
        keras.layers.BatchNormalization(name='batch_norm_1'),
        keras.layers.Dropout(0.4, name='dropout_1'),  # Increased dropout
        
        # Second hidden layer
        keras.layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_2'
        ),
        keras.layers.BatchNormalization(name='batch_norm_2'),
        keras.layers.Dropout(0.3, name='dropout_2'),
        
        # Third hidden layer (for complex patterns)
        keras.layers.Dense(
            32, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_3'
        ),
        keras.layers.Dropout(0.2, name='dropout_3'),
        
        # Output layer (Binary classification: 0=Normal, 1=Attack)
        keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile the model with lower learning rate for better generalization
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Reduced from 0.001
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
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
