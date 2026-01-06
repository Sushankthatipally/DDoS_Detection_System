"""
Federated Learning Server (Central Aggregator)
This script manages the federated learning process by:
1. Aggregating model updates from multiple clients
2. Creating a global model without seeing raw data
3. Distributing the updated model back to clients
"""

import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np

# Import metrics tracking for dashboard integration
try:
    from fl_metrics import update_server_status, update_round_metrics, reset_metrics
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Custom Federated Averaging strategy with enhanced logging
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_number = 0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate model updates from clients"""
        
        self.round_number = server_round
        
        print(f"\n{'='*70}")
        print(f"📊 ROUND {server_round} - AGGREGATING CLIENT UPDATES")
        print(f"{'='*70}")
        
        # Display client metrics
        client_metrics_list = []
        print(f"\n✅ Received updates from {len(results)} clients:")
        for i, (client, fit_res) in enumerate(results):
            metrics = fit_res.metrics
            print(f"\n   Client {i+1}:")
            print(f"      Training samples: {fit_res.num_examples:,}")
            if 'loss' in metrics:
                print(f"      Loss: {metrics['loss']:.4f}")
            if 'accuracy' in metrics:
                print(f"      Accuracy: {metrics['accuracy']:.4f}")
            
            client_metrics_list.append({
                "client_id": i,
                "samples": fit_res.num_examples,
                "loss": metrics.get('loss', 0),
                "accuracy": metrics.get('accuracy', 0)
            })
        
        if failures:
            print(f"\n⚠️  {len(failures)} clients failed to respond")
        
        # Aggregate using FedAvg algorithm
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        
        print(f"\n🔄 Global model updated using Federated Averaging")
        print(f"{'='*70}")
        
        return aggregated_weights
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ):
        """Aggregate evaluation results from clients"""
        
        print(f"\n{'='*70}")
        print(f"📈 ROUND {server_round} - EVALUATION RESULTS")
        print(f"{'='*70}")
        
        if not results:
            print("⚠️  No evaluation results received")
            return None
        
        # Calculate average metrics
        total_examples = sum([eval_res.num_examples for _, eval_res in results])
        
        # Weighted average of metrics
        weighted_loss = 0
        weighted_accuracy = 0
        weighted_precision = 0
        weighted_recall = 0
        
        print(f"\n📊 Individual Client Performance:")
        for i, (client, eval_res) in enumerate(results):
            weight = eval_res.num_examples / total_examples
            weighted_loss += eval_res.loss * weight
            
            metrics = eval_res.metrics
            if 'accuracy' in metrics:
                weighted_accuracy += metrics['accuracy'] * weight
            if 'precision' in metrics:
                weighted_precision += metrics['precision'] * weight
            if 'recall' in metrics:
                weighted_recall += metrics['recall'] * weight
            
            print(f"\n   Client {i+1}:")
            print(f"      Test samples: {eval_res.num_examples:,}")
            print(f"      Loss: {eval_res.loss:.4f}")
            if 'accuracy' in metrics:
                print(f"      Accuracy: {metrics['accuracy']:.4f}")
            if 'precision' in metrics:
                print(f"      Precision: {metrics['precision']:.4f}")
            if 'recall' in metrics:
                print(f"      Recall: {metrics['recall']:.4f}")
        
        # Display global metrics
        print(f"\n{'='*70}")
        print(f"🌍 GLOBAL MODEL PERFORMANCE (Weighted Average):")
        print(f"{'='*70}")
        print(f"   Loss: {weighted_loss:.4f}")
        print(f"   Accuracy: {weighted_accuracy:.4f}")
        print(f"   Precision: {weighted_precision:.4f}")
        print(f"   Recall: {weighted_recall:.4f}")
        f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall + 1e-7)
        print(f"   F1-Score: {f1_score:.4f}")
        print(f"{'='*70}\n")
        
        # Update metrics for dashboard
        if METRICS_ENABLED:
            try:
                global_metrics = {
                    "accuracy": weighted_accuracy,
                    "precision": weighted_precision,
                    "recall": weighted_recall,
                    "f1_score": f1_score,
                    "loss": weighted_loss
                }
                client_metrics = [{"accuracy": m.get('accuracy', 0)} for _, (_, m) in enumerate([(c, e.metrics) for c, e in results])]
                update_round_metrics(server_round, client_metrics, global_metrics)
            except Exception as e:
                print(f"⚠️  Could not update dashboard metrics: {e}")
        
        # Call parent method
        return super().aggregate_evaluate(server_round, results, failures)


def start_server(
    server_address="0.0.0.0:8080",
    num_rounds=5,
    min_clients=2
):
    """
    Start the federated learning server
    
    Args:
        server_address: Address to bind the server
        num_rounds: Number of federated learning rounds
        min_clients: Minimum number of clients required
    """
    
    print("\n" + "="*70)
    print("🚀 FEDERATED LEARNING SERVER FOR DDoS DETECTION")
    print("="*70)
    print(f"\n📡 Server Configuration:")
    print(f"   Address: {server_address}")
    print(f"   Training Rounds: {num_rounds}")
    print(f"   Minimum Clients: {min_clients}")
    print(f"\n⏳ Waiting for {min_clients} clients to connect...")
    print("="*70 + "\n")
    
    # Update metrics for dashboard
    if METRICS_ENABLED:
        try:
            reset_metrics()
            update_server_status("online", num_rounds)
            print("📊 Dashboard metrics initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize dashboard metrics: {e}")
    
    # Define federated learning strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # Use 100% of available clients for training
        fraction_evaluate=1.0,  # Use 100% of available clients for evaluation
        min_fit_clients=min_clients,  # Minimum clients for training
        min_evaluate_clients=min_clients,  # Minimum clients for evaluation
        min_available_clients=min_clients,  # Wait for this many clients to connect
    )
    
    # Start the server
    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )
        
        print("\n" + "="*70)
        print("✅ FEDERATED LEARNING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n💡 Next Steps:")
        print("   1. Check the evaluation results above")
        print("   2. The global model has learned from all clients")
        print("   3. You can now use this model for real-time DDoS detection")
        print("\n" + "="*70 + "\n")
        
        # Mark training as complete (not just offline)
        if METRICS_ENABLED:
            update_server_status("complete", num_rounds)
            print("📊 Dashboard updated: Training Complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Server stopped by user")
        if METRICS_ENABLED:
            update_server_status("offline")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        if METRICS_ENABLED:
            update_server_status("offline")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--min-clients', type=int, default=2, help='Minimum clients')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    args = parser.parse_args()
    
    start_server(
        server_address=f"0.0.0.0:{args.port}",
        num_rounds=args.rounds,
        min_clients=args.min_clients
    )
