"""
Federated Learning Client (IoT Node)
This script simulates an IoT device participating in federated learning.
Each client trains on its local data without sharing raw data with the server.
"""

import flwr as fl
import tensorflow as tf
from model import get_model
from data_utils import load_data, get_input_shape
import sys
import numpy as np

# Import metrics tracking for dashboard integration
try:
    from fl_metrics import update_client_status
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# Configuration for each client
CLIENT_CONFIGS = {
    0: {
        'name': 'IoT-Device-NTP',
        'dataset': 'C:/Users/nani/Desktop/MINOR/01-12/DrDoS_NTP.csv',
        'description': 'NTP Reflection Attack Detection'
    },
    1: {
        'name': 'IoT-Device-Portmap',
        'dataset': 'C:/Users/nani/Desktop/MINOR/03-11/Portmap.csv',
        'description': 'Portmap Attack Detection'
    },
    2: {
        'name': 'IoT-Device-DNS',
        'dataset': 'C:/Users/nani/Desktop/MINOR/01-12/DrDoS_DNS.csv',
        'description': 'DNS Reflection Attack Detection'
    }
}


class DDoSClient(fl.client.NumPyClient):
    """
    Federated Learning Client for DDoS Detection
    """
    
    def __init__(self, client_id, X_train, X_test, y_train, y_test, model):
        self.client_id = client_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.client_name = CLIENT_CONFIGS[client_id]['name']
        
        print(f"\n{'='*70}")
        print(f"🤖 {self.client_name} (Client {client_id}) Initialized")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Attack samples: {sum(y_train==1):,}")
        print(f"   Normal samples: {sum(y_train==0):,}")
        print(f"{'='*70}\n")
    
    def get_parameters(self, config):
        """Return current model parameters"""
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """
        Train the model on local data
        
        Args:
            parameters: Global model weights from server
            config: Training configuration
        
        Returns:
            Updated weights, number of samples, metrics
        """
        print(f"\n📡 {self.client_name}: Received global model from server")
        
        # Update local model with global weights
        self.model.set_weights(parameters)
        
        # Train on local data
        print(f"🔄 {self.client_name}: Training on local data...")
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=3,  # Local training epochs
            batch_size=32,
            verbose=0,  # Silent training
            validation_split=0.1
        )
        
        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        
        print(f"✅ {self.client_name}: Training complete!")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   Accuracy: {final_acc:.4f}")
        
        # Update metrics for dashboard
        if METRICS_ENABLED:
            try:
                update_client_status(self.client_id, "training", final_acc, final_loss, len(self.X_train))
            except:
                pass
        
        # Return updated weights
        return self.model.get_weights(), len(self.X_train), {
            "loss": final_loss,
            "accuracy": final_acc
        }
    
    def evaluate(self, parameters, config):
        """
        Evaluate the global model on local test data
        
        Args:
            parameters: Global model weights
            config: Evaluation configuration
        
        Returns:
            Loss, number of samples, metrics
        """
        print(f"\n🔍 {self.client_name}: Evaluating global model...")
        
        # Update model with global weights
        self.model.set_weights(parameters)
        
        # Evaluate on local test set
        loss, accuracy, precision, recall = self.model.evaluate(
            self.X_test, 
            self.y_test,
            verbose=0
        )
        
        print(f"📊 {self.client_name}: Evaluation Results:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        
        return loss, len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }


def start_client(client_id=0, server_address="127.0.0.1:8080"):
    """
    Initialize and start a federated learning client
    
    Args:
        client_id: ID of the client (0, 1, or 2)
        server_address: Address of the federated server
    """
    print(f"\n{'='*70}")
    print(f"🚀 STARTING FEDERATED LEARNING CLIENT {client_id}")
    print(f"{'='*70}")
    
    # Update DATASET_PATH in data_utils
    import data_utils
    data_utils.DATASET_PATH = CLIENT_CONFIGS[client_id]['dataset']
    
    print(f"\n📂 Loading dataset: {CLIENT_CONFIGS[client_id]['dataset']}")
    print(f"   Description: {CLIENT_CONFIGS[client_id]['description']}")
    
    # Load data for this client
    X_train, X_test, y_train, y_test = load_data(client_id=0, num_clients=1)
    
    if X_train is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Create model
    input_shape = get_input_shape()
    model = get_model(input_shape)
    
    # Create client instance
    client = DDoSClient(client_id, X_train, X_test, y_train, y_test, model)
    
    # Connect to server and start federated learning
    print(f"\n🌐 Connecting to server at {server_address}...")
    print("⏳ Waiting for training instructions from server...\n")
    
    # Update metrics for dashboard
    if METRICS_ENABLED:
        try:
            update_client_status(client_id, "online", 0, 0, len(X_train))
        except:
            pass
    
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
    except Exception as e:
        print(f"\n❌ Error connecting to server: {e}")
        print("💡 Make sure the server is running first!")
        if METRICS_ENABLED:
            try:
                update_client_status(client_id, "offline")
            except:
                pass


if __name__ == "__main__":
    # Get client ID from command line argument
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        print("Usage: python client.py <client_id>")
        print("Example: python client.py 0")
        print("\nAvailable clients:")
        for cid, config in CLIENT_CONFIGS.items():
            print(f"  {cid}: {config['name']} - {config['description']}")
        sys.exit(1)
    
    if client_id not in CLIENT_CONFIGS:
        print(f"❌ Invalid client ID: {client_id}")
        print(f"Available client IDs: {list(CLIENT_CONFIGS.keys())}")
        sys.exit(1)
    
    # Start the client
    start_client(client_id=client_id)
