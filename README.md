# Multimodal-30-day-readmission-prediction-using-State-Space-Models
Multimodal 30-Day Readmission Prediction with State Space Models

Predicting 30-day hospital readmission for adult ICU patients from full-stay ICU time series and clinical notes, using a Mamba (selective state space model) encoder fused with clinical language model note embeddings.

Fall 2026 semester project, Southern Methodist University.

Team: Balakrishnan Arumugam, Siddharth Rohitkumar Sheth, Krishna Vipulkumar Patel

Motivation

Most EHR readmission models truncate the ICU stay to a fixed window (commonly the first 24-48 hours) because recurrent and attention-based models scale poorly with long sequences. State space models such as Mamba scale linearly in sequence length, which makes it practical to encode the entire stay at hourly resolution. This project tests whether that longer context, combined with information from clinical notes, improves readmission prediction over standard baselines.

Task
Cohort: adult patients (age >= 18) with at least one ICU stay in MIMIC-IV
Prediction point: hospital discharge
Label: unplanned readmission within 30 days of discharge
Inputs: hourly ICU time series (vitals, labs, interventions) and clinical notes from the index admission
Data
MIMIC-IV (hosp and icu modules)
MIMIC-IV-Note (discharge summaries and radiology reports)

Both datasets are credentialed-access on PhysioNet. This repository contains no patient data. To reproduce:

Complete PhysioNet credentialing and sign the data use agreement for MIMIC-IV and MIMIC-IV-Note.
Download the data to a location on SMU HPC.
Set MIMIC_ROOT and MIMIC_NOTE_ROOT in configs/paths.yaml.

Per the DUA, all processing runs locally on SMU HPC. Notes are not sent to external APIs or hosted LLM services.

Method
Time series encoder: Mamba over the full stay, binned to 1-hour resolution with missingness masks and time-since-last-observation features
Note encoder: frozen clinical language model, producing note-level embeddings that are pooled per admission
Fusion: concatenation of the time series and note representations, followed by a classification head
Concept features: UMLS concepts extracted from notes, used for an auxiliary analysis and interpretation
Baselines
LACE index
Logistic regression and XGBoost on aggregated tabular features
GRU
GRU-D
Transformer encoder
Experiments
Main comparison: SSM multimodal model vs. baselines
Context-length ablation: first 48 hours vs. first 7 days vs. full stay
Modality ablation: time series only, notes only, both
Interpretation: integrated gradients over time steps and input features
Evaluation: AUROC, AUPRC, calibration (ECE, reliability curves), subgroup performance, and temporal validation (train on earlier admissions, test on later ones)
Repository Structure
configs/            paths and experiment configs
data/               cohort definitions and preprocessing scripts (no data)
src/
  cohort/           cohort selection and label construction
  features/         time series binning and tabular feature extraction
  notes/            note cleaning, embedding, UMLS extraction
  models/           Mamba encoder, fusion model, baselines
  train.py
  evaluate.py
slurm/              job scripts for SMU HPC
notebooks/          exploratory analysis
results/            metrics and figures (generated)
Setup
conda create -n ssm-ehr python=3.10
conda activate ssm-ehr
pip install -r requirements.txt

mamba-ssm requires a CUDA-capable GPU and a compatible CUDA toolkit. Build it on a GPU node, not the login node.

Running on SMU HPC
sbatch slurm/preprocess.sh
sbatch slurm/embed_notes.sh
sbatch slurm/train.sh configs/ssm_full_stay.yaml

Jobs are capped at 4 hours, so training scripts checkpoint regularly and can resume from the latest checkpoint.

Status

Proposal stage. Cohort extraction and preprocessing are in progress.

Acknowledgments

MIMIC-IV and MIMIC-IV-Note are provided by the MIT Laboratory for Computational Physiology through PhysioNet.
