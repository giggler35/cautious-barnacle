"""
disease_codes.py

Single source of truth for the 4-digit disease codes, mapped to the
21 class names in the exact order the model was trained with
(sorted(df["label"].unique())). Every script -- server, app, teammates'
code -- should reference this instead of retyping the list, so codes
never get out of sync between pieces built by different people.
"""

DISEASE_CODES = {
    "1000": "Acne and Rosacea Photos",
    "1001": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "1002": "Atopic Dermatitis Photos",
    "1003": "Bullous Disease Photos",
    "1004": "Cellulitis Impetigo and other Bacterial Infections",
    "1005": "Eczema Photos",
    "1006": "Exanthems and Drug Eruptions",
    "1007": "Hair Loss Photos Alopecia and other Hair Diseases",
    "1008": "Herpes HPV and other STDs Photos",
    "1009": "Light Diseases and Disorders of Pigmentation",
    "1010": "Melanoma Skin Cancer Nevi and Moles",
    "1011": "Nail Fungus and other Nail Disease",
    "1012": "Poison Ivy Photos and other Contact Dermatitis",
    "1013": "Psoriasis pictures Lichen Planus and related diseases",
    "1014": "Scabies Lyme Disease and other Infestations and Bites",
    "1015": "Seborrheic Keratoses and other Benign Tumors",
    "1016": "Systemic and Autoimmune Conditions",
    "1017": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "1018": "Urticaria Hives",
    "1019": "Vascular Tumors",
    "1020": "Warts Molluscum and other Viral Infections",
}

# Reverse lookup: class name -> code
NAME_TO_CODE = {v: k for k, v in DISEASE_CODES.items()}

# Same order the model outputs predictions in (sorted class names)
CLASS_NAMES_SORTED = sorted(DISEASE_CODES.values())
