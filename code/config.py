"""
config.py
Central configuration for the multi-label 10-K classifier project.
All other scripts import paths and constants from here.
"""

from pathlib import Path

# Project-root directory (determined automatically; no hard-coded drive letters)
# ROOT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = "../"
# TEMP_DIR = "~/temp/"
TEMP_DIR = "../temp/"

# Key sub-directories
CODE  = ROOT_DIR + "code/"
DATA  = ROOT_DIR + "data/"
MODEL = TEMP_DIR + "models/"

CODE_DIR  = Path(CODE)
DATA_DIR  = Path(DATA)
MODEL_DIR = Path(MODEL)

# Data locations (relative paths keep the repo portable)
PROCESSED_TRAIN_DF = DATA + "training_df.csv"
INFERENCE_TEXT_DIR = Path(DATA + "item_1c/")

# Output tables
MODEL_COMPARISON_CSV   = MODEL + "model_comparison_results.csv"
CHECKPOINT_METRICS_CSV = MODEL + "model_checkpoint_metrics.csv"
INFERENCE_OUT_CSV      = DATA + "item_1c_inference.csv"
INFERENCE_PARTIAL_CSV  = DATA + "item_1c_inference_partial.csv"

# Claude:
"""
I. The SEC-Mandated Baseline (4 Required Categories)
These are the four disclosure areas formally required by SEC Regulation S-K Item 106, and form the minimum coding framework for any systematic analysis:

Risk identification and assessment processes — Companies share the frequency, scope, and internal distribution of their risk assessments. The SEC requires this disclosure. Hyperproof
Risk management practices — Companies detail their methods for addressing identified risks, often including their approach to determining risk impact and likelihood, using equal measures of qualitative and quantitative methods. Hyperproof
Material incident disclosure — whether prior cybersecurity incidents have materially affected or are reasonably likely to affect the registrant (directly relevant to your research question about previously attacked firms).
Governance — board oversight and management's role and expertise in cybersecurity risk management.


II. Extended Categories Observed in Empirical Studies (up to 11 Topics)
Practitioner research reviewing over 2,000+ filings found that over 40% of companies are sharing additional information on eleven different topics, going beyond the four mandated disclosures. Hyperproof The additional categories that have been coded in the literature include:

Cybersecurity framework adoption — 51% of companies reference one or more cybersecurity frameworks, attestations, or external regulatory requirements as the foundation of their cybersecurity program, likely as a proxy for showing due care. Hyperproof NIST CSF and ISO 27001 are the most commonly cited.
Third-party and supply chain risk management — Nearly all companies (98%) generally disclose the engagement of assessors, consultants, auditors or other third parties in the management of cybersecurity risks. Harvard Law School Forum on Corporate Governance Whether companies have specific processes to oversee third-party service providers is coded as a distinct dimension.
ERM integration — Analysis of Item 1C has been conducted to determine whether each firm's cybersecurity risk management (CRM) system was integrated within the firm's enterprise risk management (ERM) framework, coded as a binary indicator. ResearchGate This has been linked empirically to material weaknesses in internal controls.
Incident response and recovery plans — Companies are disclosing collaboration with external partners to enhance cybersecurity measures, existence of response and recovery plans and playbooks, and steps for conducting tests and recovery approaches. PwC
Security awareness and training programs — Some companies specified how cyber risk is prioritized across the organization through training and awareness programs. PwC
Management expertise and CISO credentials — disclosure of specific qualifications, certifications (e.g., CISSP), prior roles, and years of experience of the responsible officer.
Governance structure and board oversight specifics — Governance is often an entire subsection in Section 1C, detailing how companies internally manage cybersecurity-related processes and identify process owners, disclosing quarterly cybersecurity risk assessments and board communication of these assessments. Hyperproof
"""

LABEL_COLS = [
    "risk_mgmt", # Quantifies the threat and provide guidance to the firm on how much protection it needs
    "governance", # System of rules, structures, and decision-making authority that sets direction and defines values
    "compliance", # Aligns day-to-day operations with external laws (like GDPR, HIPAA, or SEC regulations) and internal governance policies
    "oversight", # Operational function that checks and monitors whether those rules and goals are actually being followed
]

# Shared training hyperparameters
NUM_EPOCHS        = 4
TRAIN_BATCH_SIZE  = 4
EVAL_BATCH_SIZE   = 4
LOGGING_STEPS     = 50
METRIC_KEY        = "f1_micro"
SAVE_TOTAL_LIMIT  = 1   # keep only the latest checkpoint per model

# Inference-time settings
CHUNK_SIZE         = 510     # special tokens add to 512
BATCH_SIZE         = 1
AGGREGATION_METHOD = "max"
THRESHOLD          = 0.5