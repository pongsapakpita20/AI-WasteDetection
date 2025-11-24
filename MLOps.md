## AI Waste Sorter – MLOps Guide

### DVC Pipeline
- **Config** lives in `params.yaml`. Adjust training/eval hyperparameters here.
- Run full pipeline locally:
  ```
  dvc pull waste-detection.dvc   # fetch dataset from remote
  dvc repro evaluate             # runs train -> promote_best -> evaluate
  dvc push                       # upload artifacts/metrics to remote
  ```
- Stage outputs:
  - `artifacts/models/waste-sorter-best.pt` (latest promoted weights, tracked via DVC `outs`)
  - `artifacts/eval/metrics.json` (evaluation summary, tracked as DVC metrics)
- Update/initialize a remote (example):
  ```
  dvc remote add -d storage s3://<bucket>/ai-waste-sorter
  dvc remote modify storage access_key_id <key>
  dvc remote modify storage secret_access_key <secret>
  ```

### Airflow Orchestration
- DAG file: `airflow/dags/ai_waste_sorter_dag.py`
  - `pull_dvc_data` → `run_dvc_pipeline` → `push_dvc_artifacts` → `notify_success`
  - Uses repository root inferred from the DAG location.
- Quick start (Docker Compose):
  1. Copy repo into Airflow deployment (e.g., mount at `/opt/airflow/dags/..`).
  2. Ensure `dvc`, `git`, `python`, and GPU drivers (if needed) are installed inside the Airflow worker image.
  3. Provide credentials for the configured DVC remote (ENV vars or `.dvc/config`).
  4. Enable the DAG from the Airflow UI and monitor runs/metrics there.

### Git Workflow
- Commit the following (already prepared):
  - `dvc.yaml`, `params.yaml`, updated `train.py`, `evaluate.py`, `tools/promote_best.py`
  - `.gitignore` updates, `airflow/dags/ai_waste_sorter_dag.py`, `MLOps.md`
- Large artifacts (`waste-detection`, `artifacts/`, `runs/`, `.dvc/cache/`) stay out of git and are versioned by DVC instead.

### Next Steps
- Configure a DVC remote backend that suits your infra (S3, GDrive, Azure, etc.).
- Hook Airflow DAG notifications into Slack/Email if needed.
- Optionally extend DAG with separate staging/production deploy tasks once automated deployment scripts are ready.

