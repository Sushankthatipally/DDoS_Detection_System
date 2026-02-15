# Federated Learning for Decentralized DDoS Attack Detection in IoT Networks

## Abstract
Write the final abstract here using clear, source-aligned wording.
Do not include uncited quantitative claims.

## Problem Definition
Describe:
- IoT DDoS threat model
- Limitations of centralized detection
- Privacy, bandwidth, latency, and scalability constraints

## System Architecture
Document the implemented architecture using:
- `server.py` as central aggregator
- `client.py` as distributed training nodes
- `fl_metrics.py` and `fl_metrics.json` for metric persistence
- `dashboard.py` for monitoring

Include a simple data flow:
1. Server distributes global model.
2. Clients train locally.
3. Clients send weight updates only.
4. Server aggregates updates with FedAvg.

## Data Preprocessing
Base this section on `data_utils.py`:
- Dataset ingestion
- Feature cleanup and leakage column removal
- Label encoding (`BENIGN -> 0`, attack -> 1)
- Scaling and split strategy
- Mixed-attack data construction logic

## Model and Training Design
Base this section on `model.py` and `client.py`:
- Neural network architecture
- Loss and optimizer
- Evaluation metrics
- Early stopping
- Class weighting

## Federated Workflow
Document operational steps and control parameters:
- Server startup arguments (`--rounds`, `--min-clients`, `--port`)
- Client IDs and startup command pattern
- Round-based training and evaluation cycle

## Experimental Setup
Capture environment details:
- Python version
- Package versions
- Hardware summary (if available)
- Dataset file paths used for the run
- Round/client configuration

## Results (Evidence-Backed Only)
Rule:
- Every number must include source path and timestamp.
- If source is missing, mark as `Pending measurement`.

### Evidence Sources
- `fl_metrics.json`
- Saved terminal logs from server/client runs
- Generated plots or exported figures

### Results Table
| Metric | Value | Source Path | Timestamp | State |
| --- | --- | --- | --- | --- |
| Global accuracy | Pending measurement | Pending measurement | Pending measurement | Pending measurement |
| Global precision | Pending measurement | Pending measurement | Pending measurement | Pending measurement |
| Global recall | Pending measurement | Pending measurement | Pending measurement | Pending measurement |
| Global F1-score | Pending measurement | Pending measurement | Pending measurement | Pending measurement |

## Limitations
Track known constraints such as:
- Hard-coded absolute dataset paths
- Potential client data-loading inconsistency
- Demo values in dashboard not equal to measured evidence
- Environment compatibility limits

## Future Work
List prioritized follow-up items, for example:
1. Path configuration via environment variables or config files.
2. Client-specific data partition correctness fix.
3. Automated experiment logging and artifact management.
4. Robust evaluation across additional attack families and non-IID distributions.

## References
Add all literature, dataset links, and library references used in the report.

