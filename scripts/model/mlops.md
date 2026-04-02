Here's a prioritized MLOps roadmap for DeepOMAPNet, based on what currently exists and what's
  missing:                                                                 
                                                                                                
  ---                                              
  Current MLOps State                                                                           
                                                                                                
  Has: Basic GitHub Actions CI, pytest suite (50+ tests), synthetic data fixtures, mixed        
  precision training, early stopping.                                                           
                                                                                                
  Missing: Experiment tracking, logging, config management, Docker, model registry, data        
  versioning, monitoring.                                                                       
                                                                                                
  ---                                                                                           
  Recommended Work by Priority                                                                  
                              
  Tier 1 — High impact, low effort (do first)                                                   
                                                                                                
  1. Experiment Tracking (wandb or MLflow)
  - The training loop in gat_trainer.py returns metrics as a plain dict. Wrap it with           
  wandb/MLflow to log loss curves, hyperparameters, and model artifacts automatically.          
  - This unblocks reproducibility across team members and datasets.                   
                                                                                                
  2. Config-driven hyperparameters                                                              
  - Training config is hardcoded in notebooks (cell 9 in Training.ipynb). Move to a config.yaml
  parsed at runtime. This is the single biggest friction point when running sweeps.             
                  
  3. Structured logging                                                                         
  - Replace print() statements in gat_trainer.py with Python logging module — file + console
  handlers. Enables post-hoc debugging on long runs.                                            
                                                    
  4. Code quality in CI                                                                         
  - ci.yml installs black/flake8 but never runs them. Add a lint step. Add --cov to pytest for  
  coverage reporting.                                                                           
                                                                                                
  ---                                                                                           
  Tier 2 — Medium effort, enables scale
                                                                                                
  5. Containerization (Docker)
  - PyTorch Geometric has notoriously finicky install (CUDA-version-specific wheels). A         
  Dockerfile eliminates this for every collaborator and any cloud GPU setup.                    
  
  6. Data versioning (DVC)                                                                      
  - .gitignore explicitly excludes *.h5ad data files. Right now there's no reproducible way to
  track which dataset version produced which model. DVC + S3/GCS remote fixes this.             
  
  7. Model registry                                                                             
  - Currently model.pth is saved manually. Add metadata (training config, eval metrics, dataset
  hash, git commit SHA) alongside each checkpoint. MLflow Model Registry or even a structured   
  JSON manifest works.
                                                                                                
  8. Hyperparameter sweeps
  - The model has ~8 tunable hyperparameters. Add wandb Sweeps or Optuna integration on top of
  run_experiment.py — it already has the right structure for this.                              
  
  ---                                                                                           
  Tier 3 — Longer term, production readiness
                                                                                                
  9. Pipeline orchestration
  - The data flow (normalize → build graph → train → evaluate → visualize) is notebook-driven.  
  Refactor run_experiment.py into a Prefect or Metaflow DAG to enable resumable, parallelizable 
  runs on large datasets.
                                                                                                
  10. Scalable inference
  - SparseCrossAttentionLayer already handles large graphs. Add a standalone inference.py with
  batched prediction, ONNX export option, and a simple REST endpoint (FastAPI) for serving      
  trained models.
                                                                                                
  11. Data quality monitoring
  - Single-cell data has batch effects and dropout. Add distribution shift detection (e.g.,
  KL-divergence on RNA/ADT distributions) before training to catch bad input data early.        
  
  12. GPU cluster training                                                                      
  - The trainer supports single-GPU. Add PyTorch DDP or Lightning for multi-GPU, and integrate
  with SLURM job submission (common in bioinformatics HPC environments).                        
  
  ---                                                                                           
  Suggested Execution Order
                           
  Week 1-2:  Config YAML + structured logging + wandb integration
  Week 3-4:  Dockerfile + DVC + model registry metadata                                         
  Month 2:   Hyperparameter sweeps + CI code quality
  Month 3+:  Prefect DAG + distributed training + inference API                                 
                  
  The highest-leverage single item is experiment tracking — without it, every training run is a 
  black box and you can't systematically improve the model. Everything else builds on top of
  that.                                                                                         
                  
  Want me to implement any of these? The config YAML + wandb integration is the best place to   
  start.