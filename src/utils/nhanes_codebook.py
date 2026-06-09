"""NHANES variable code → human-readable label mapping."""

CODEBOOK: dict[str, str] = {
    # Identifiers
    "SEQN": "Respondent Sequence Number",
    # Demographics (P_DEMO)
    "RIDAGEYR": "Age (years)",
    "RIAGENDR": "Sex (1=Male, 2=Female)",
    "RIDRETH3": "Race/Ethnicity",
    "WTMECPRP": "MEC Exam Sample Weight",
    "SDMVPSU": "Masked Variance Pseudo-PSU",
    "SDMVSTRA": "Masked Variance Pseudo-Stratum",
    # FibroScan / LUX (P_LUX)
    "LUAXSTAT": "Elastography Exam Status",
    "LUXSMED": "Median Liver Stiffness (kPa)",
    "LUXSIQR": "Liver Stiffness IQR (kPa)",
    "LUXSIQRM": "Liver Stiffness IQR/Median (%)",
    "LUXCAPM": "Median CAP (dB/m)",
    "LUXCPIQR": "CAP IQR (dB/m)",
    "LUANMVGP": "Number of Complete Stiffness Measures",
    "LUAPNME": "Wand Type (M or XL)",
    # Biochemistry (P_BIOPRO)
    "LBXSATSI": "ALT (U/L)",
    "LBXSASSI": "AST (U/L)",
    "LBXSGTSI": "GGT (U/L)",
    "LBXSAL": "Albumin (g/dL)",
    "LBXSAPSI": "ALP (U/L)",
    "LBXSTB": "Total Bilirubin (mg/dL)",
    "LBXSGL": "Serum Glucose (mg/dL)",
    "LBXSCR": "Creatinine (mg/dL)",
    "LBXSBU": "BUN (mg/dL)",
    "LBXSUA": "Uric Acid (mg/dL)",
    "LBXSPH": "Phosphorus (mg/dL)",
    # CBC (P_CBC)
    "LBXPLTSI": "Platelet Count (1000 cells/μL)",
    "LBXHGB": "Haemoglobin (g/dL)",
    "LBXWBCSI": "WBC (1000 cells/μL)",
    "LBXRBCSI": "RBC (million cells/μL)",
    "LBXMCVSI": "MCV (fL)",
    "LBXRDW": "RDW (%)",
    "LBXMPSI": "Mean Platelet Volume (fL)",
    "LBXNEPCT": "Neutrophil % (%)",
    "LBXLYPCT": "Lymphocyte % (%)",
    # Glycohemoglobin (P_GHB)
    "LBXGH": "HbA1c (%)",
    # Body measures (P_BMX)
    "BMXBMI": "BMI (kg/m²)",
    "BMXWAIST": "Waist Circumference (cm)",
    "BMXWT": "Weight (kg)",
    "BMXHT": "Height (cm)",
    # Lipids (P_TRIGLY, P_HDL, P_TCHOL)
    "LBXTR": "Triglycerides (mg/dL)",
    "LBXLDL": "LDL Cholesterol (mg/dL)",
    "LBDHDD": "HDL Cholesterol (mg/dL)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    # Diabetes (P_DIQ)
    "DIQ010": "Diabetes Diagnosis (1=Yes, 2=No, 3=Borderline)",
    "DIQ160": "Prediabetes",
    # Alcohol (P_ALQ)
    "ALQ111": "Had 12+ Drinks Ever",
    "ALQ121": "Frequency of Alcohol Use",
    "ALQ130": "Avg Drinks/Day Past 12 Months",
    # Blood pressure (P_BPX)
    "BPXOSY1": "Systolic BP Reading 1 (mmHg)",
    "BPXODI1": "Diastolic BP Reading 1 (mmHg)",
    # Hepatitis (P_HEQ)
    "HEQ010": "History of Hepatitis B",
    "HEQ030": "History of Hepatitis C",
    # Smoking (P_SMQ)
    "SMQ020": "Smoked ≥100 Cigarettes",
    "SMQ040": "Current Smoker",
    # Derived scores
    "FIB4": "FIB-4 Index",
    "FIB4_CAT": "FIB-4 Category (0=Low, 1=Indeterminate, 2=High)",
    "NFS": "NAFLD Fibrosis Score",
    "APRI": "AST-to-Platelet Ratio Index",
    "DE_RITIS": "AST/ALT Ratio (de Ritis)",
    "HSI": "Hepatic Steatosis Index",
    "SII": "Systemic Immune-Inflammation Index",
    "TYG": "Triglyceride-Glucose Index",
    "EGFR": "Estimated GFR (CKD-EPI)",
    "WHTR": "Waist-to-Height Ratio",
    "LSM_CAT": "LSM Category (0=No/Mild, 1=Significant, 2=Advanced)",
    "FIB4_FALSE_NEGATIVE": "FIB-4 False Negative (1=Yes)",
    "FIB4_FALSE_POSITIVE": "FIB-4 False Positive (1=Yes)",
    "MASLD_ELIGIBLE": "MASLD Eligibility",
    "MISCLASS_LABEL": "Misclassification Label",
}

# Race/ethnicity decode
RACE_LABELS: dict[int, str] = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multi-Racial",
}

# Right-skewed variables that should be log-transformed
LOG_TRANSFORM_VARS = [
    "LBXSATSI", "LBXSASSI", "LBXSGTSI", "LBXTR", "LBXSTB",
    "LBXSAPSI", "LBXSUA", "SII", "TYG",
]

# All feature variables for modelling (excluding target and identifiers)
MODEL_FEATURES = [
    # FIB-4 native
    "RIDAGEYR", "LBXSASSI", "LBXSATSI", "LBXPLTSI",
    # Extended biochemistry
    "LBXSGTSI", "LBXSAL", "LBXSAPSI", "LBXSTB", "LBXSCR",
    "LBXSBU", "LBXSGL", "LBXSUA",
    # Metabolic
    "LBXGH", "LBXTR", "LBDHDD", "LBXLDL", "LBXTC", "TYG",
    # Anthropometric
    "BMXBMI", "BMXWAIST", "WHTR",
    # Hematological
    "LBXHGB", "LBXWBCSI", "LBXMCVSI", "LBXRDW", "LBXMPSI",
    # Inflammatory / derived
    "SII", "DE_RITIS",
    # Derived scores (not including FIB-4 itself as it's the grouping criterion)
    "NFS", "APRI", "HSI",
]
