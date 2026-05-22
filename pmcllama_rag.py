#!/usr/bin/env python3
"""PMC-LLaMA 13B + BioLORD retrieval. Watch ctx — only 2048 tokens."""


from __future__ import annotations
import os
import re
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CSV_DIR = Path("/workspace/LLM_research/treatRag/csv_files")
OUT_DIR = Path("/workspace/LLM_research/treatRag/output_pmcllama_rag")
CHROMA_DB_DIR = OUT_DIR / "chroma_db"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_RATIO = 0.8
MAX_PROMPT_TOKENS = 1800       # 2048 ctx; reserve for generation
MAX_NEW_TOKENS = 128
LLM_MODEL = "axiong/PMC_LLaMA_13B"
BATCH_SIZE = 4                 # 13B; tight on VRAM

EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_NORMALIZE = True
CHROMA_COLLECTION_NAME = "treatrag_pmcllama_biolord"
CHROMA_DISTANCE_METRIC = "cosine"

RAG_TOP_K = 30
RAG_MAX_CASES = 10
RAG_MIN_SIMILARITY = 0.3
RAG_THRESHOLD_MULT = 1.5

# shared drug -> atc3 dict
DRUG_TO_ATC3: Dict[str, str] = {
    # A - alimentary
    "omeprazole":       "A02B", "pantoprazole":     "A02B", "lansoprazole":     "A02B",
    "esomeprazole":     "A02B", "famotidine":       "A02B", "ranitidine":       "A02B",
    "protonix":         "A02B", "nexium":           "A02B", "prilosec":         "A02B",
    "pepcid":           "A02B",
    "sucralfate":       "A02B", "carafate":         "A02B",
    "metoclopramide":   "A03F", "reglan":           "A03F",
    "ondansetron":      "A04A", "granisetron":      "A04A", "zofran":           "A04A",
    "prochlorperazine": "A04A",
    "docusate":         "A06A", "senna":            "A06A", "sennosides":       "A06A",
    "bisacodyl":        "A06A", "polyethylene glycol": "A06A", "lactulose":     "A06A",
    "miralax":          "A06A", "colace":           "A06A", "dulcolax":         "A06A",
    "sodium phosphate": "A06A",
    "insulin":          "A10A", "insulin regular":  "A10A", "insulin glargine": "A10A",
    "insulin lispro":   "A10A", "insulin aspart":   "A10A", "insulin nph":      "A10A",
    "humulin":          "A10A", "novolog":          "A10A", "lantus":           "A10A",
    "metformin":        "A10B", "glipizide":        "A10B", "glyburide":        "A10B",
    "sitagliptin":      "A10B", "pioglitazone":     "A10B", "glucophage":       "A10B",
    "multivitamin":     "A11A", "thiamine":         "A11D", "vitamin b1":       "A11D",
    "folic acid":       "A11B", "folate":           "A11B",
    "pyridoxine":       "A11H", "vitamin b6":       "A11H",
    "ascorbic acid":    "A11G", "vitamin c":        "A11G",
    "cholecalciferol":  "A11C", "ergocalciferol":   "A11C", "vitamin d":        "A11C",
    "calcium carbonate":"A12A", "calcium gluconate": "A12A", "calcium chloride": "A12A",
    "calcium":          "A12A",
    "potassium chloride":"A12B","potassium":        "A12B", "potassium phosphate":"A12B",
    "magnesium sulfate":"A12C", "magnesium oxide":  "A12C", "magnesium":        "A12C",
    "magnesium hydroxide":"A12C","zinc sulfate":    "A12C", "zinc":             "A12C",
    "sodium bicarbonate":"B05X", "sodium chloride": "B05B",
    "phosphorus":       "A12C", "sodium phosphates":"A12C",

    # B - blood
    "heparin":          "B01A", "enoxaparin":       "B01A", "lovenox":          "B01A",
    "aspirin":          "B01A", "clopidogrel":      "B01A", "plavix":           "B01A",
    "warfarin":         "B01A", "coumadin":         "B01A",
    "alteplase":        "B01A", "ticagrelor":       "B01A", "bivalirudin":      "B01A",
    "argatroban":       "B01A", "fondaparinux":     "B01A", "prasugrel":        "B01A",
    "rivaroxaban":      "B01A", "apixaban":         "B01A", "dabigatran":       "B01A",
    "dalteparin":       "B01A",
    "ferrous sulfate":  "B03A", "iron sucrose":     "B03A", "iron dextran":     "B03A",
    "epoetin":          "B03X", "darbepoetin":      "B03X",
    "albumin":          "B05A", "albumin human":    "B05A",
    "dextrose":         "B05B", "lactated ringers":  "B05B", "normal saline":   "B05B",
    "d5w":              "B05B", "d5ns":             "B05B",
    "phytonadione":     "B02B", "vitamin k":        "B02B", "tranexamic acid":  "B02B",
    "aminocaproic acid":"B02A",
    "protamine":        "V03A",

    # C - cardio
    "digoxin":          "C01A", "lanoxin":          "C01A",
    "amiodarone":       "C01B", "lidocaine":        "C01B", "procainamide":     "C01B",
    "flecainide":       "C01B", "sotalol":          "C01B",
    "dobutamine":       "C01C", "milrinone":        "C01C",
    "norepinephrine":   "C01C", "levophed":         "C01C",
    "epinephrine":      "C01C", "adrenaline":       "C01C",
    "phenylephrine":    "C01C", "neosynephrine":    "C01C",
    "vasopressin":      "C01C", "dopamine":         "C01C",
    "nitroglycerin":    "C01D", "isosorbide":       "C01D", "nitroprusside":    "C01D",
    "clonidine":        "C02A", "hydralazine":      "C02D",
    "furosemide":       "C03C", "lasix":            "C03C", "bumetanide":       "C03C",
    "bumex":            "C03C", "torsemide":        "C03C", "ethacrynic acid":  "C03C",
    "spironolactone":   "C03D", "aldactone":        "C03D", "eplerenone":       "C03D",
    "hydrochlorothiazide":"C03A","chlorothiazide":  "C03A", "metolazone":       "C03B",
    "metoprolol":       "C07A", "atenolol":         "C07A", "propranolol":      "C07A",
    "labetalol":        "C07A", "esmolol":          "C07A", "carvedilol":       "C07A",
    "bisoprolol":       "C07A", "nadolol":          "C07A", "lopressor":        "C07A",
    "coreg":            "C07A",
    "amlodipine":       "C08C", "diltiazem":        "C08C", "nifedipine":       "C08C",
    "nicardipine":      "C08C", "verapamil":        "C08C", "cardizem":         "C08C",
    "norvasc":          "C08C", "clevidipine":      "C08C",
    "lisinopril":       "C09A", "enalapril":        "C09A", "captopril":        "C09A",
    "ramipril":         "C09A", "benazepril":       "C09A", "quinapril":        "C09A",
    "losartan":         "C09C", "valsartan":        "C09C", "irbesartan":       "C09C",
    "olmesartan":       "C09C", "candesartan":      "C09C",
    "atorvastatin":     "C10A", "simvastatin":      "C10A", "rosuvastatin":     "C10A",
    "pravastatin":      "C10A", "lovastatin":       "C10A", "lipitor":          "C10A",
    "crestor":          "C10A", "fluvastatin":      "C10A",

    # D - derm
    "mupirocin":        "D06A", "bactroban":        "D06A",
    "nystatin":         "D01A",

    # G - GU
    "tamsulosin":       "G04C", "flomax":           "G04C", "finasteride":      "G04C",
    "oxybutynin":       "G04B",

    # H - hormones
    "hydrocortisone":   "H02A", "methylprednisolone":"H02A","dexamethasone":    "H02A",
    "prednisone":       "H02A", "prednisolone":     "H02A", "solumedrol":       "H02A",
    "decadron":         "H02A", "cortisol":         "H02A", "fludrocortisone":  "H02A",
    "budesonide":       "H02A",
    "levothyroxine":    "H03A", "synthroid":        "H03A", "liothyronine":     "H03A",

    # J - antimicrobials
    "ampicillin":       "J01C", "amoxicillin":      "J01C", "piperacillin":     "J01C",
    "piperacillin-tazobactam":"J01C", "zosyn":      "J01C", "nafcillin":        "J01C",
    "oxacillin":        "J01C", "unasyn":           "J01C",
    "ampicillin-sulbactam":"J01C", "amoxicillin-clavulanate":"J01C", "augmentin":"J01C",
    "penicillin":       "J01C", "ticarcillin":      "J01C", "dicloxacillin":    "J01C",
    "ceftriaxone":      "J01D", "cefazolin":        "J01D", "cefepime":         "J01D",
    "ceftazidime":      "J01D", "cefoxitin":        "J01D", "cefuroxime":       "J01D",
    "cephalexin":       "J01D", "cefdinir":         "J01D", "ancef":            "J01D",
    "rocephin":         "J01D", "maxipime":         "J01D", "cefpodoxime":      "J01D",
    "ceftaroline":      "J01D",
    "meropenem":        "J01D", "imipenem":         "J01D", "ertapenem":        "J01D",
    "doripenem":        "J01D",
    "azithromycin":     "J01F", "erythromycin":     "J01F", "clarithromycin":   "J01F",
    "zithromax":        "J01F",
    "gentamicin":       "J01G", "tobramycin":       "J01G", "amikacin":         "J01G",
    "ciprofloxacin":    "J01M", "levofloxacin":     "J01M", "moxifloxacin":     "J01M",
    "cipro":            "J01M", "levaquin":         "J01M",
    "vancomycin":       "J01X", "metronidazole":    "J01X", "flagyl":           "J01X",
    "linezolid":        "J01X", "daptomycin":       "J01X", "clindamycin":      "J01X",
    "trimethoprim":     "J01E", "sulfamethoxazole":  "J01E", "bactrim":         "J01E",
    "trimethoprim-sulfamethoxazole":"J01E", "nitrofurantoin": "J01X",
    "doxycycline":      "J01A", "tetracycline":     "J01A", "minocycline":      "J01A",
    "tigecycline":      "J01A", "colistin":         "J01X", "polymyxin":        "J01X",
    "fluconazole":      "J02A", "micafungin":       "J02A", "caspofungin":      "J02A",
    "anidulafungin":    "J02A", "voriconazole":     "J02A", "amphotericin":     "J02A",
    "amphotericin b":   "J02A", "itraconazole":     "J02A", "posaconazole":     "J02A",
    "diflucan":         "J02A",
    "acyclovir":        "J05A", "valacyclovir":     "J05A", "ganciclovir":      "J05A",
    "oseltamivir":      "J05A", "tamiflu":          "J05A", "valganciclovir":   "J05A",

    # L - immuno (transplant pts)
    "tacrolimus":       "L04A", "mycophenolate":    "L04A", "cyclosporine":     "L04A",
    "sirolimus":        "L04A", "azathioprine":     "L04A", "basiliximab":      "L04A",

    # M - MSK + NMBAs
    "ibuprofen":        "M01A", "ketorolac":        "M01A", "naproxen":         "M01A",
    "toradol":          "M01A", "meloxicam":        "M01A", "celecoxib":        "M01A",
    "indomethacin":     "M01A", "diclofenac":       "M01A",
    "baclofen":         "M03B", "cyclobenzaprine":  "M03B", "tizanidine":       "M03B",
    "methocarbamol":    "M03B",
    "cisatracurium":    "M03A", "rocuronium":       "M03A", "vecuronium":       "M03A",
    "succinylcholine":  "M03A", "pancuronium":      "M03A", "atracurium":       "M03A",

    # N - CNS / sedation
    "propofol":         "N01A", "ketamine":         "N01A", "etomidate":        "N01A",
    "sevoflurane":      "N01A", "desflurane":       "N01A", "diprivan":         "N01A",
    "thiopental":       "N01A",
    "morphine":         "N02A", "fentanyl":         "N02A", "hydromorphone":    "N02A",
    "oxycodone":        "N02A", "codeine":          "N02A", "meperidine":       "N02A",
    "remifentanil":     "N02A", "sufentanil":       "N02A", "methadone":        "N02A",
    "tramadol":         "N02A", "dilaudid":         "N02A", "oxycontin":        "N02A",
    "buprenorphine":    "N02A", "nalbuphine":       "N02A",
    "acetaminophen":    "N02B", "tylenol":          "N02B",
    "levetiracetam":    "N03A", "phenytoin":        "N03A", "valproic acid":    "N03A",
    "carbamazepine":    "N03A", "gabapentin":       "N03A", "pregabalin":       "N03A",
    "keppra":           "N03A", "dilantin":         "N03A", "depakote":         "N03A",
    "lacosamide":       "N03A", "oxcarbazepine":    "N03A", "topiramate":       "N03A",
    "lamotrigine":      "N03A", "phenobarbital":    "N03A", "divalproex":       "N03A",
    "haloperidol":      "N05A", "quetiapine":       "N05A", "olanzapine":       "N05A",
    "risperidone":      "N05A", "aripiprazole":     "N05A", "haldol":           "N05A",
    "seroquel":         "N05A", "ziprasidone":      "N05A", "chlorpromazine":   "N05A",
    "lorazepam":        "N05B", "midazolam":        "N05B", "diazepam":         "N05B",
    "alprazolam":       "N05B", "ativan":           "N05B", "versed":           "N05B",
    "clonazepam":       "N05B",
    "zolpidem":         "N05C", "ambien":           "N05C", "melatonin":        "N05C",
    "dexmedetomidine":  "N05C", "precedex":         "N05C",
    "sertraline":       "N06A", "fluoxetine":       "N06A", "citalopram":       "N06A",
    "escitalopram":     "N06A", "trazodone":        "N06A", "mirtazapine":      "N06A",
    "paroxetine":       "N06A", "venlafaxine":      "N06A", "duloxetine":       "N06A",
    "bupropion":        "N06A", "amitriptyline":    "N06A", "nortriptyline":    "N06A",
    "lexapro":          "N06A", "zoloft":           "N06A", "prozac":           "N06A",
    "wellbutrin":       "N06A", "effexor":          "N06A", "cymbalta":         "N06A",
    "donepezil":        "N06D", "memantine":        "N06D",
    "naloxone":         "V03A", "narcan":           "V03A", "flumazenil":       "V03A",
    "neostigmine":      "N07A", "pyridostigmine":   "N07A",
    "methylphenidate":  "N06B", "dextroamphetamine":"N06B", "modafinil":        "N06B",

    # P - antiparasitic
    "hydroxychloroquine":"P01B",

    # R - respiratory
    "albuterol":        "R03A", "ipratropium":      "R03B", "tiotropium":       "R03B",
    "salbutamol":       "R03A", "levalbuterol":     "R03A", "proventil":        "R03A",
    "ventolin":         "R03A", "combivent":        "R03A", "atrovent":         "R03B",
    "fluticasone":      "R03B", "beclomethasone":   "R03B", "mometasone":       "R03B",
    "montelukast":      "R03D", "singulair":        "R03D", "theophylline":     "R03D",
    "aminophylline":    "R03D",
    "guaifenesin":      "R05C", "dextromethorphan": "R05D", "benzonatate":      "R05D",
    "diphenhydramine":  "R06A", "cetirizine":       "R06A", "hydroxyzine":      "R06A",
    "loratadine":       "R06A", "fexofenadine":     "R06A", "benadryl":         "R06A",
    "promethazine":     "R06A", "zyrtec":           "R06A", "claritin":         "R06A",
    "phenylephrine nasal":"R01A", "oxymetazoline":  "R01A",

    # V - misc / reversal
    "mannitol":         "B05B",
    "acetylcysteine":   "V03A", "mucomyst":         "V03A",
    "deferoxamine":     "V03A",
    "sugammadex":       "V03A",
}

def normalize_drug_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\d+\.?\d*\s*(mg|mcg|ml|g|units?|%|meq)\b.*", "", name)
    name = re.sub(r"\s*(tablet|capsule|injection|solution|suspension|cream|"
                  r"ointment|syrup|patch|suppository|inhaler|vial|bag|"
                  r"powder|liquid|drops|spray|gel|lotion|iv|oral|topical|"
                  r"ophthalmic|otic|nasal|rectal|sublingual|transdermal)\b.*", "", name)
    name = re.sub(r"\s*\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def map_drug_to_atc3(drug_name: str) -> Optional[str]:
    normalized = normalize_drug_name(drug_name)
    if not normalized:
        return None
    if normalized in DRUG_TO_ATC3:
        return DRUG_TO_ATC3[normalized]
    first_word = normalized.split()[0] if normalized else ""
    if first_word and len(first_word) > 2 and first_word in DRUG_TO_ATC3:
        return DRUG_TO_ATC3[first_word]
    for known_drug, atc3 in DRUG_TO_ATC3.items():
        if len(known_drug) > 3 and known_drug in normalized:
            return atc3
    for known_drug, atc3 in DRUG_TO_ATC3.items():
        if len(known_drug) > 3 and normalized.startswith(known_drug):
            return atc3
    return None


def load_and_preprocess_data() -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 70)

    logger.info("Loading CSV files...")
    patients = pd.read_csv(CSV_DIR / "patients.csv")
    admissions = pd.read_csv(CSV_DIR / "admissions.csv")
    prescriptions = pd.read_csv(CSV_DIR / "prescriptions.csv",
                                 usecols=["subject_id", "hadm_id", "drug", "ndc"])
    drgcodes = pd.read_csv(CSV_DIR / "drgcodes.csv",
                            usecols=["subject_id", "hadm_id", "drg_type", "description"])
    d_icd = pd.read_csv(CSV_DIR / "d_icd_diagnoses.csv")

    logger.info(f"  Patients:      {len(patients):,}")
    logger.info(f"  Admissions:    {len(admissions):,}")
    logger.info(f"  Prescriptions: {len(prescriptions):,}")
    logger.info(f"  DRG codes:     {len(drgcodes):,}")

    logger.info("Building diagnosis map from DRG codes...")
    drg_diag = drgcodes[drgcodes["drg_type"] == "APR"].copy()
    if len(drg_diag) == 0:
        drg_diag = drgcodes.copy()
    drg_diag["description"] = drg_diag["description"].fillna("").astype(str)
    diag_per_adm = (
        drg_diag.groupby(["subject_id", "hadm_id"])["description"]
        .apply(lambda x: list(x.unique()))
        .reset_index()
        .rename(columns={"description": "diagnoses"})
    )
    logger.info(f"  Diagnosis records: {len(diag_per_adm):,}")

    logger.info("Mapping drugs to ATC-3 codes...")
    prescriptions["drug"] = prescriptions["drug"].fillna("").astype(str)
    prescriptions["atc3"] = prescriptions["drug"].apply(map_drug_to_atc3)

    total_rx = len(prescriptions)
    mapped_rx = prescriptions["atc3"].notna().sum()
    logger.info(f"  Drug mapping: {mapped_rx:,}/{total_rx:,} ({100*mapped_rx/total_rx:.1f}%)")

    rx_mapped = prescriptions[prescriptions["atc3"].notna()].copy()
    atc3_per_adm = (
        rx_mapped.groupby(["subject_id", "hadm_id"])["atc3"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
        .rename(columns={"atc3": "atc3_codes"})
    )
    logger.info(f"  Admissions with ATC-3 codes: {len(atc3_per_adm):,}")

    drug_names_per_adm = (
        rx_mapped.groupby(["subject_id", "hadm_id"])["drug"]
        .apply(lambda x: list(x.unique())[:10])
        .reset_index()
        .rename(columns={"drug": "drug_names"})
    )

    logger.info("Merging admission data...")
    adm = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm = adm.sort_values(["subject_id", "admittime"])

    adm = adm.merge(diag_per_adm, on=["subject_id", "hadm_id"], how="left")
    adm = adm.merge(atc3_per_adm, on=["subject_id", "hadm_id"], how="left")
    adm = adm.merge(drug_names_per_adm, on=["subject_id", "hadm_id"], how="left")

    adm["diagnoses"] = adm["diagnoses"].apply(lambda x: x if isinstance(x, list) else [])
    adm["atc3_codes"] = adm["atc3_codes"].apply(lambda x: x if isinstance(x, list) else [])
    adm["drug_names"] = adm["drug_names"].apply(lambda x: x if isinstance(x, list) else [])

    adm = adm.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")

    adm["has_diag"] = adm["diagnoses"].apply(lambda x: len(x) > 0)
    adm["has_rx"] = adm["atc3_codes"].apply(lambda x: len(x) > 0)
    logger.info(f"  Total admissions: {len(adm):,}")
    logger.info(f"  With diagnoses:   {adm['has_diag'].sum():,}")
    logger.info(f"  With ATC-3 codes: {adm['has_rx'].sum():,}")

    adm["visit_num"] = adm.groupby("subject_id").cumcount() + 1
    adm["total_visits"] = adm.groupby("subject_id")["hadm_id"].transform("count")

    logger.info(f"  Unique patients: {adm['subject_id'].nunique():,}")
    logger.info(f"  Avg visits/patient: {adm['total_visits'].mean():.2f}")

    out_path = OUT_DIR / "preprocessed_admissions.csv"
    save_df = adm.copy()
    save_df["diagnoses"] = save_df["diagnoses"].apply(json.dumps)
    save_df["atc3_codes"] = save_df["atc3_codes"].apply(json.dumps)
    save_df["drug_names"] = save_df["drug_names"].apply(json.dumps)
    save_df.to_csv(out_path, index=False)
    logger.info(f"  Saved preprocessed data -> {out_path}")

    all_atc3 = [code for codes in adm["atc3_codes"] for code in codes]
    atc3_counts = pd.Series(all_atc3).value_counts()
    logger.info(f"\n  Unique ATC-3 codes: {len(atc3_counts)}")
    logger.info(f"  Top 10 ATC-3 codes:\n{atc3_counts.head(10).to_string()}")

    return adm


def build_patient_prompt(visits: pd.DataFrame, include_last_rx: bool = False,
                         max_history_visits: int = 4,
                         use_drug_names: bool = False) -> str:
    total = len(visits)
    if total > max_history_visits + 1:
        visits = visits.iloc[-(max_history_visits + 1):]

    actual = len(visits)
    parts = [f"The patient has {total} times ICU visits."]

    ordinals = ["first", "second", "third", "fourth", "fifth",
                "sixth", "seventh", "eighth", "ninth", "tenth"]

    for i, (_, v) in enumerate(visits.iterrows()):
        is_last = (i == actual - 1)
        ord_str = ordinals[min(i, len(ordinals)-1)] if i < len(ordinals) else f"{i+1}th"

        diag_list = v["diagnoses"] if isinstance(v["diagnoses"], list) else []
        diag_str = ", ".join(diag_list[:3]) if diag_list else "unspecified"

        if is_last and not include_last_rx:
            parts.append(
                f"In this visit, the patient has diagnosis: {diag_str}."
            )
            parts.append("Then, the patient should be prescribed:")
        else:
            if use_drug_names:
                drug_list = v["drug_names"] if isinstance(v["drug_names"], list) else []
                rx_str = ", ".join(drug_list[:6]) if drug_list else "none recorded"
            else:
                atc3_list = v["atc3_codes"] if isinstance(v["atc3_codes"], list) else []
                rx_str = ", ".join(atc3_list[:6]) if atc3_list else "none recorded"

            parts.append(
                f"In the {ord_str} visit, the patient had diagnosis: {diag_str}. "
                f"The patient was prescribed: {rx_str}."
            )

    return " ".join(parts)


def build_patient_text_for_retrieval(visits: pd.DataFrame) -> str:
    parts = []
    for _, v in visits.iterrows():
        diag_list = v["diagnoses"] if isinstance(v["diagnoses"], list) else []
        atc3_list = v["atc3_codes"] if isinstance(v["atc3_codes"], list) else []
        if diag_list:
            parts.append("diagnosis: " + ", ".join(diag_list[:5]))
        if atc3_list:
            parts.append("prescribed: " + ", ".join(atc3_list[:8]))
    return " ".join(parts)


class BioLORDRetrievalIndex:
    def __init__(self, reset_db: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL} on {device}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        test_emb = self.embedding_model.encode("test", normalize_embeddings=EMBEDDING_NORMALIZE)
        self.embedding_dim = len(test_emb)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        if reset_db:
            try:
                self.client.delete_collection(name=CHROMA_COLLECTION_NAME)
                logger.info(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
        )
        logger.info(f"ChromaDB ready: {self.collection.count():,} documents, "
                     f"embedding dim={self.embedding_dim}")

    def add_documents(self, documents: List[Dict], show_progress: bool = True) -> None:
        if not documents:
            return
        texts = [d["text"] for d in documents]
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=EMBEDDING_NORMALIZE,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        ids = [str(d["pid"]) for d in documents]
        metadatas = [
            {
                "pid": int(d["pid"]),
                "atc3_codes": json.dumps(d["atc3_codes"]),
                "prompt_text": d["prompt_text"][:500],
            }
            for d in documents
        ]
        CHROMA_BATCH = 5000
        for i in tqdm(range(0, len(documents), CHROMA_BATCH),
                      desc="Adding to ChromaDB", disable=not show_progress):
            j = min(i + CHROMA_BATCH, len(documents))
            self.collection.add(
                ids=ids[i:j],
                embeddings=embeddings[i:j].tolist(),
                documents=texts[i:j],
                metadatas=metadatas[i:j],
            )
        logger.info(f"Indexed {len(documents):,} documents into ChromaDB")

    def query_batch(
        self,
        query_texts: List[str],
        query_pids: List[int],
        n_results: int = RAG_TOP_K,
    ) -> List[List[Dict]]:
        query_pid_set = set(query_pids)

        all_embeddings = self.embedding_model.encode(
            query_texts,
            normalize_embeddings=EMBEDDING_NORMALIZE,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        all_results = []
        RETRIEVAL_BATCH = 32

        for batch_start in tqdm(range(0, len(query_texts), RETRIEVAL_BATCH),
                                desc="Batch embedding retrieval"):
            batch_end = min(batch_start + RETRIEVAL_BATCH, len(query_texts))
            batch_embs = all_embeddings[batch_start:batch_end]

            results = self.collection.query(
                query_embeddings=batch_embs.tolist(),
                n_results=n_results * 2,
                include=["metadatas", "distances"],
            )

            for i in range(len(batch_embs)):
                retrieved_cases = []
                for md, dist in zip(results["metadatas"][i], results["distances"][i]):
                    retrieved_pid = int(md["pid"])
                    if retrieved_pid in query_pid_set:
                        continue
                    sim = 1.0 - float(dist)
                    if sim >= RAG_MIN_SIMILARITY:
                        retrieved_cases.append({
                            "pid": retrieved_pid,
                            "similarity": sim,
                            "atc3_codes": json.loads(md["atc3_codes"]),
                            "prompt_text": md.get("prompt_text", ""),
                        })

                retrieved_cases.sort(key=lambda x: x["similarity"], reverse=True)
                all_results.append(retrieved_cases)

        return all_results


def apply_adaptive_gap_threshold(
    cases: List[Dict],
    multiplier: float = RAG_THRESHOLD_MULT,
) -> List[Dict]:
    if len(cases) <= 1:
        return cases
    gaps = [cases[i - 1]["similarity"] - cases[i]["similarity"] for i in range(1, len(cases))]
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps) if len(gaps) > 1 else 0.0
    threshold = mean_gap + multiplier * std_gap

    kept = [cases[0]]
    for i in range(1, len(cases)):
        gap = kept[-1]["similarity"] - cases[i]["similarity"]
        if gap > threshold:
            break
        kept.append(cases[i])
    return kept


def build_llm_prompt_no_rag(patient_prompt: str) -> str:
    return patient_prompt


def build_llm_prompt_with_rag(
    patient_prompt: str,
    similar_cases: List[str],
) -> str:
    prompt_parts = [
        "ICU medication prediction. List ATC-3 codes, comma-separated.\n"
    ]

    if similar_cases:
        prompt_parts.append("Similar cases:")
        for i, case in enumerate(similar_cases, 1):
            prompt_parts.append(f"{i}. {case[:200]}")
        prompt_parts.append("")

    prompt_parts.append(patient_prompt)

    return "\n".join(prompt_parts)


ATC3_PATTERN = re.compile(r"\b([A-Z]\d{2}[A-Z])\b")


def extract_atc3_from_output(
    raw_output: str,
    valid_atc3_codes: Set[str],
    strict: bool = False,
) -> List[str]:
    if not raw_output:
        return []

    text = raw_output.strip()
    found_codes: Set[str] = set()

    for match in ATC3_PATTERN.finditer(text.upper()):
        code = match.group(1)
        if code in valid_atc3_codes:
            found_codes.add(code)

    if strict:
        items = re.split(r"[,;|\n]", text)
        for item in items:
            item = item.strip().strip(".").strip()
            if not item or len(item) < 3:
                continue
            normalized = normalize_drug_name(item)
            if normalized in DRUG_TO_ATC3:
                found_codes.add(DRUG_TO_ATC3[normalized])
    else:
        text_lower = text.lower()
        for drug_name, atc3 in DRUG_TO_ATC3.items():
            if len(drug_name) > 3 and drug_name in text_lower:
                found_codes.add(atc3)

        if not found_codes:
            items = re.split(r"[,;|\n]", text)
            for item in items:
                item = item.strip().strip(".")
                if not item:
                    continue
                upper_item = item.upper().strip()
                if ATC3_PATTERN.match(upper_item) and upper_item in valid_atc3_codes:
                    found_codes.add(upper_item)
                atc3 = map_drug_to_atc3(item)
                if atc3:
                    found_codes.add(atc3)

    return sorted(found_codes)


def compute_f1_jaccard(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float]:
    if not true_set and not pred_set:
        return 1.0, 1.0
    if not true_set or not pred_set:
        return 0.0, 0.0

    intersection = true_set & pred_set
    if not intersection:
        return 0.0, 0.0

    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(true_set)
    f1 = 2 * precision * recall / (precision + recall)
    jaccard = len(intersection) / len(true_set | pred_set)

    return f1, jaccard


def evaluate_predictions(
    results: List[Dict],
    label: str,
) -> Dict[str, float]:
    f1_scores = []
    jaccard_scores = []
    n_total = 0
    n_nonempty_true = 0
    n_nonempty_pred = 0

    for r in results:
        true_set = set(r["true_atc3"])
        pred_set = set(r["pred_atc3"])

        if true_set:
            n_nonempty_true += 1
        if pred_set:
            n_nonempty_pred += 1
        n_total += 1

        if true_set:
            f1, jacc = compute_f1_jaccard(true_set, pred_set)
            f1_scores.append(f1)
            jaccard_scores.append(jacc)

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    avg_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATION RESULTS: {label}")
    logger.info(f"{'='*70}")
    logger.info(f"  Total patients evaluated: {n_total}")
    logger.info(f"  Patients with ground truth: {n_nonempty_true}")
    logger.info(f"  Patients with predictions:  {n_nonempty_pred}")
    logger.info(f"  F1-score:           {avg_f1:.4f}")
    logger.info(f"  Jaccard Similarity: {avg_jaccard:.4f}")
    logger.info(f"{'='*70}\n")

    return {
        "method": label,
        "n_total": n_total,
        "n_with_true": n_nonempty_true,
        "n_with_pred": n_nonempty_pred,
        "f1_score": round(avg_f1, 4),
        "jaccard_similarity": round(avg_jaccard, 4),
    }


def run_llm_inference(
    prompts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> List[str]:
    all_outputs = []

    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="PMC-LLaMA inference"):
        batch_prompts = prompts[start_idx:start_idx + batch_size]

        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = encoded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True,
        )
        all_outputs.extend(outputs)

    return all_outputs


def main():
    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected — inference will be slow!")

    adm = load_and_preprocess_data()

    logger.info("=" * 70)
    logger.info("STEP 2: BUILDING PATIENT STRUCTURES")
    logger.info("=" * 70)

    patient_groups = dict(list(adm.groupby("subject_id")))
    eligible_patients = {}

    for pid, group in patient_groups.items():
        group = group.sort_values("admittime").reset_index(drop=True)
        if len(group) < 2:
            continue
        last_visit = group.iloc[-1]
        if not last_visit["diagnoses"] or len(last_visit["diagnoses"]) == 0:
            continue
        if not last_visit["atc3_codes"] or len(last_visit["atc3_codes"]) == 0:
            continue
        prior = group.iloc[:-1]
        if not any(len(codes) > 0 for codes in prior["atc3_codes"]):
            continue
        eligible_patients[pid] = group

    logger.info(f"  Eligible patients (>=2 visits with data): {len(eligible_patients):,}")

    all_pids = sorted(eligible_patients.keys())
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_pids)

    split_idx = int(len(all_pids) * TRAIN_RATIO)
    train_pids = all_pids[:split_idx]
    test_pids = all_pids[split_idx:]

    logger.info(f"  Train patients: {len(train_pids):,}")
    logger.info(f"  Test patients:  {len(test_pids):,}")

    t_preprocess = time.time()

    logger.info("Building text representations for train patients...")
    train_texts: Dict[int, str] = {}
    train_prompts_full: Dict[int, str] = {}

    for pid in tqdm(train_pids, desc="Train patient texts"):
        group = eligible_patients[pid]
        text = build_patient_text_for_retrieval(group)
        train_texts[pid] = text
        train_prompts_full[pid] = build_patient_prompt(group, include_last_rx=True, use_drug_names=False)

    all_atc3_set: Set[str] = set()
    for pid in all_pids:
        for _, row in eligible_patients[pid].iterrows():
            if isinstance(row["atc3_codes"], list):
                all_atc3_set.update(row["atc3_codes"])
    logger.info(f"  Total unique ATC-3 codes in data: {len(all_atc3_set)}")

    logger.info("=" * 70)
    logger.info("STEP 3: BUILDING BioLORD-2023 EMBEDDING INDEX")
    logger.info("=" * 70)

    retrieval_index = BioLORDRetrievalIndex(reset_db=True)

    index_documents = []
    for pid in train_pids:
        group = eligible_patients[pid]
        last_visit = group.iloc[-1]
        atc3_codes = last_visit["atc3_codes"] if isinstance(last_visit["atc3_codes"], list) else []
        index_documents.append({
            "pid": pid,
            "text": train_texts[pid],
            "atc3_codes": atc3_codes,
            "prompt_text": train_prompts_full[pid],
        })

    retrieval_index.add_documents(index_documents)

    t_embed = time.time()

    logger.info("Building test patient prompts...")
    test_data = []
    for pid in tqdm(test_pids, desc="Test patient prompts"):
        group = eligible_patients[pid]
        last_visit = group.iloc[-1]

        true_atc3 = last_visit["atc3_codes"] if isinstance(last_visit["atc3_codes"], list) else []

        baseline_prompt = build_patient_prompt(group, include_last_rx=False, use_drug_names=True)
        rag_prompt = build_patient_prompt(group, include_last_rx=False, use_drug_names=False)

        query_text = build_patient_text_for_retrieval(group.iloc[:-1])
        diag_list = last_visit["diagnoses"] if isinstance(last_visit["diagnoses"], list) else []
        if diag_list:
            query_text += " diagnosis: " + ", ".join(diag_list[:5])

        test_data.append({
            "pid": pid,
            "baseline_prompt": baseline_prompt,
            "rag_prompt": rag_prompt,
            "true_atc3": true_atc3,
            "query_text": query_text,
        })

    logger.info(f"  Test samples: {len(test_data):,}")

    logger.info("=" * 70)
    logger.info("LOADING PMC-LLaMA 13B MODEL")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    logger.info(f"  Model loaded on {device}")

    t_model_load = time.time()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE B: TreatRAG PMC-LLaMA 13B (With BioLORD-2023 Embedding RAG)")
    logger.info("=" * 70)

    query_texts = [td["query_text"] for td in test_data]
    test_pid_list = [td["pid"] for td in test_data]

    all_retrieved = retrieval_index.query_batch(
        query_texts, test_pid_list, n_results=RAG_TOP_K,
    )

    rag_prompts = []
    retrieval_stats = {"total": 0, "with_cases": 0, "avg_cases": []}

    for td, retrieved_cases in zip(test_data, all_retrieved):
        retrieval_stats["total"] += 1

        filtered_cases = retrieved_cases[:RAG_MAX_CASES]
        filtered_cases = apply_adaptive_gap_threshold(filtered_cases, RAG_THRESHOLD_MULT)

        if filtered_cases:
            retrieval_stats["with_cases"] += 1
            retrieval_stats["avg_cases"].append(len(filtered_cases))

            case_texts = []
            for case in filtered_cases[:5]:
                prompt_text = case.get("prompt_text", "")
                if prompt_text:
                    case_texts.append(f"(sim:{case['similarity']:.2f}) {prompt_text[:200]}")

            prompt = build_llm_prompt_with_rag(td["rag_prompt"], case_texts)
        else:
            prompt = build_llm_prompt_no_rag(td["baseline_prompt"])

        rag_prompts.append(prompt)

    logger.info(f"\n  Retrieval stats:")
    logger.info(f"    Total queries:       {retrieval_stats['total']:,}")
    logger.info(f"    With similar cases:  {retrieval_stats['with_cases']:,} "
                 f"({100*retrieval_stats['with_cases']/max(1,retrieval_stats['total']):.1f}%)")
    if retrieval_stats["avg_cases"]:
        logger.info(f"    Avg cases retrieved: {np.mean(retrieval_stats['avg_cases']):.1f}")

    logger.info(f"\n--- Sample RAG prompt (first patient) ---")
    logger.info(rag_prompts[0][:800])
    logger.info("---\n")

    t_retrieval = time.time()

    rag_outputs = run_llm_inference(
        rag_prompts, tokenizer, model, device, BATCH_SIZE,
    )

    rag_results = []
    for td, raw_output in zip(test_data, rag_outputs):
        pred_atc3 = extract_atc3_from_output(raw_output, all_atc3_set, strict=False)
        rag_results.append({
            "pid": td["pid"],
            "true_atc3": td["true_atc3"],
            "pred_atc3": pred_atc3,
            "raw_output": raw_output[:300],
        })

    for i in range(min(5, len(rag_results))):
        r = rag_results[i]
        logger.info(f"  Patient {r['pid']}: "
                     f"true={r['true_atc3'][:5]} pred={r['pred_atc3'][:5]} "
                     f"raw='{r['raw_output'][:100]}'")

    t_inference = time.time()

    rag_metrics = evaluate_predictions(rag_results, "TreatRAG PMC-LLaMA 13B (BioLORD-2023 RAG)")

    rag_df = pd.DataFrame([{
        "subject_id": r["pid"],
        "true_atc3": json.dumps(r["true_atc3"]),
        "pred_atc3": json.dumps(r["pred_atc3"]),
        "raw_output": r["raw_output"],
    } for r in rag_results])
    rag_df.to_csv(OUT_DIR / "rag_results.csv", index=False)

    t_evaluation = time.time()

    logger.info(f"\n{'='*70}")
    logger.info("TIME COST BREAKDOWN")
    logger.info(f"{'='*70}")
    logger.info(f"  Data preprocessing:      {t_preprocess - t_start:.1f}s")
    logger.info(f"  Embedding index:         {t_embed - t_preprocess:.1f}s")
    logger.info(f"  Model loading:           {t_model_load - t_embed:.1f}s")
    logger.info(f"  Retrieval:               {t_retrieval - t_model_load:.1f}s")
    logger.info(f"  Inference:               {t_inference - t_retrieval:.1f}s")
    logger.info(f"  Evaluation:              {t_evaluation - t_inference:.1f}s")
    logger.info(f"  Total:                   {t_evaluation - t_start:.1f}s ({(t_evaluation - t_start)/60:.1f} min)")

    summary = {
        "rag_metrics": rag_metrics,
        "dataset_stats": {
            "total_patients": len(all_pids),
            "train_patients": len(train_pids),
            "test_patients": len(test_pids),
            "unique_atc3_codes": len(all_atc3_set),
        },
        "config": {
            "model": "PMC-LLaMA-13B",
            "embedding_model": EMBEDDING_MODEL,
            "rag_top_k": RAG_TOP_K,
            "rag_max_cases": RAG_MAX_CASES,
            "rag_min_similarity": RAG_MIN_SIMILARITY,
            "random_state": RANDOM_STATE,
        },
        "time_cost": {
            "data_preprocessing_sec": round(t_preprocess - t_start, 2),
            "embedding_index_building_sec": round(t_embed - t_preprocess, 2),
            "model_loading_sec": round(t_model_load - t_embed, 2),
            "retrieval_sec": round(t_retrieval - t_model_load, 2),
            "inference_sec": round(t_inference - t_retrieval, 2),
            "evaluation_sec": round(t_evaluation - t_inference, 2),
            "total_sec": round(t_evaluation - t_start, 2),
            "total_minutes": round((t_evaluation - t_start) / 60, 2),
        },
    }

    with open(OUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {OUT_DIR}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
