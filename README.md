# CGMPOC_Pipeline — Interpretable Machine Learning for Postprandial Glucose Prediction

This repository contains a **proof-of-concept pipeline** demonstrating how interpretable machine learning (ML) can be used to model and explain **postprandial glucose responses (PPGR)** from multimodal health data.  
The project was developed as part of a **Master of Research (MRes)** at the *University of Salford* (2026–2027).

---

## Overview

The pipeline demonstrates a fully modular workflow for:

1. **Data ingestion and cleaning** of continuous glucose monitor (CGM) and lifestyle records  
2. **Feature engineering** — daily metrics, per-meal outcomes, and contextual variables  
3. **Interpretable ML models** — Ridge Regression, Generalized Additive Models (GAMs), and Decision Trees  
4. **Explainability (XAI)** — SHAP-based feature attributions and simple decision rules  
5. **Reproducible reporting** — automated HTML summaries and visual diagnostics  

All models are validated using **Leave-One-Participant-Out (LOPO)** cross-validation to ensure generalization to unseen individuals.

---

## Project Structure

```bash
scripts/       → analysis modules (01–11)
src/           → configuration + helpers
reports/       → generated summaries (not included)
data/          → structure retained, no data files committed
requirements.txt
README.md
```
## Data Privacy
No participant-level or institutional data are stored in this repository.
Only the pipeline code and directory structure are shared to support reproducible research while respecting GDPR and ethical requirements.
This pipeline can be executed on any similarly structured, anonymized dataset.

## Environment
Developed with Python 3.9+.
To install dependencies: 
```bash
pip install -r requirements.txt
```
## Planned Extensions
• Integration of Gut Microbiome + Clinical Biomarkers 

•	Bayesian and SHAP-based explainability layers

•	Cross-dataset generalization and personalization studies

## Author

Niloofar Ebrahimi

Email: s.niloofar.ebrahimi@gmail.com