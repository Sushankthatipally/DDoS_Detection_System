"""
Shared metrics storage for Federated Learning Dashboard
This module provides file-based communication between FL components and dashboard
"""

import json
import os
from datetime import datetime
from pathlib import Path

METRICS_FILE = Path(__file__).parent / "fl_metrics.json"

def get_default_metrics():
    """Return default metrics structure"""
    return {
        "server_status": "offline",
        "training_complete": False,
        "server_started_at": None,
        "training_completed_at": None,
        "current_round": 0,
        "total_rounds": 5,
        "clients": {
            "0": {"status": "offline", "name": "IoT-Device-NTP", "accuracy": 0, "loss": 0, "samples": 0, "training_complete": False},
            "1": {"status": "offline", "name": "IoT-Device-Portmap", "accuracy": 0, "loss": 0, "samples": 0, "training_complete": False},
            "2": {"status": "offline", "name": "IoT-Device-DNS", "accuracy": 0, "loss": 0, "samples": 0, "training_complete": False}
        },
        "global_metrics": {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "loss": 0
        },
        "round_history": [],
        "attack_log": [],
        "last_updated": None
    }

def load_metrics():
    """Load metrics from file"""
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, 'r') as f:
                return json.load(f)
        except:
            return get_default_metrics()
    return get_default_metrics()

def save_metrics(metrics):
    """Save metrics to file"""
    metrics["last_updated"] = datetime.now().isoformat()
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

def update_server_status(status, num_rounds=5):
    """Update server status"""
    metrics = load_metrics()
    metrics["server_status"] = status
    metrics["total_rounds"] = num_rounds
    if status == "online":
        metrics["server_started_at"] = datetime.now().isoformat()
        metrics["training_complete"] = False
    elif status == "complete":
        metrics["server_status"] = "offline"
        metrics["training_complete"] = True
        metrics["training_completed_at"] = datetime.now().isoformat()
        # Mark all clients as training complete
        for client_id in metrics["clients"]:
            if metrics["clients"][client_id]["accuracy"] > 0:
                metrics["clients"][client_id]["training_complete"] = True
    save_metrics(metrics)

def update_client_status(client_id, status, accuracy=0, loss=0, samples=0):
    """Update client status"""
    metrics = load_metrics()
    client_id = str(client_id)
    if client_id in metrics["clients"]:
        metrics["clients"][client_id]["status"] = status
        if accuracy > 0:
            metrics["clients"][client_id]["accuracy"] = accuracy
        if loss > 0:
            metrics["clients"][client_id]["loss"] = loss
        if samples > 0:
            metrics["clients"][client_id]["samples"] = samples
        if status == "complete":
            metrics["clients"][client_id]["training_complete"] = True
            metrics["clients"][client_id]["status"] = "offline"
    save_metrics(metrics)

def update_round_metrics(round_num, client_metrics, global_metrics):
    """Update metrics after a round"""
    metrics = load_metrics()
    metrics["current_round"] = round_num
    metrics["global_metrics"] = global_metrics
    
    # Add to history
    metrics["round_history"].append({
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "global_accuracy": global_metrics.get("accuracy", 0),
        "global_loss": global_metrics.get("loss", 0),
        "client_metrics": client_metrics
    })
    
    save_metrics(metrics)

def add_attack_log(source_ip, attack_type, confidence, status):
    """Add an attack detection to the log"""
    metrics = load_metrics()
    metrics["attack_log"].insert(0, {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source_ip": source_ip,
        "attack_type": attack_type,
        "confidence": confidence,
        "status": status
    })
    # Keep only last 50 entries
    metrics["attack_log"] = metrics["attack_log"][:50]
    save_metrics(metrics)

def reset_metrics():
    """Reset all metrics to default"""
    save_metrics(get_default_metrics())

def get_training_status():
    """Get current training status"""
    metrics = load_metrics()
    
    # Count clients based on their state
    active_clients = sum(1 for c in metrics["clients"].values() if c["status"] == "online")
    completed_clients = sum(1 for c in metrics["clients"].values() if c.get("training_complete", False))
    
    return {
        "is_training": metrics["server_status"] == "online",
        "training_complete": metrics.get("training_complete", False),
        "current_round": metrics["current_round"],
        "total_rounds": metrics["total_rounds"],
        "active_clients": active_clients,
        "completed_clients": completed_clients
    }
