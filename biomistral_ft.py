#!/usr/bin/env python3
# QLoRA (4-bit) fine-tune for BioMistral-7B, then inference. SKIP_TRAIN=1 to reload an adapter.

from __future__ import annotations
import os
import re
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CSV_DIR = Path("/workspace/LLM_research/treatRag_MIMIC_IV/csv_files")
OUT_DIR = Path("/workspace/LLM_research/treatRag_MIMIC_IV/output_biomistral_ft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1            # 70/10/20, matches GRU baseline

MODEL = "BioMistral/BioMistral-7B"
MAX_SEQ_LEN = 2048
MAX_PROMPT_TOKENS = 1800
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4

# LoRA / QLoRA knobs
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
FT_EPOCHS = 5                # bumped from 3 to match eICU run
FT_BATCH_SIZE = 2
FT_GRAD_ACCUM = 16
FT_LR = 1e-4                 # half of 2e-4 — diverged at 2e-4 w/ 4-bit + LoRA on all 7 modules
FT_WARMUP_RATIO = 0.1
FT_MAX_GRAD_NORM = 0.5       # NaN-fix; see INFERENCE_FIX_NOTES.md

# SKIP_TRAIN=1 -> skip training, just reload adapter for inference
SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "0").lower() in ("1", "true", "yes")
ADAPTER_DIR_NAME = "lora_adapter"



# shared drug -> atc3 dict
DRUG_TO_ATC3: Dict[str, str] = {
    # A
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
    # B
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
    # C
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
    # D
    "mupirocin":        "D06A", "bactroban":        "D06A",
    "nystatin":         "D01A",
    # G
    "tamsulosin":       "G04C", "flomax":           "G04C", "finasteride":      "G04C",
    "oxybutynin":       "G04B",
    # H
    "hydrocortisone":   "H02A", "methylprednisolone":"H02A","dexamethasone":    "H02A",
    "prednisone":       "H02A", "prednisolone":     "H02A", "solumedrol":       "H02A",
    "decadron":         "H02A", "cortisol":         "H02A", "fludrocortisone":  "H02A",
    "budesonide":       "H02A",
    "levothyroxine":    "H03A", "synthroid":        "H03A", "liothyronine":     "H03A",
    # J
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
    # L
    "tacrolimus":       "L04A", "mycophenolate":    "L04A", "cyclosporine":     "L04A",
    "sirolimus":        "L04A", "azathioprine":     "L04A", "basiliximab":      "L04A",
    # M
    "ibuprofen":        "M01A", "ketorolac":        "M01A", "naproxen":         "M01A",
    "toradol":          "M01A", "meloxicam":        "M01A", "celecoxib":        "M01A",
    "indomethacin":     "M01A", "diclofenac":       "M01A",
    "baclofen":         "M03B", "cyclobenzaprine":  "M03B", "tizanidine":       "M03B",
    "methocarbamol":    "M03B",
    "cisatracurium":    "M03A", "rocuronium":       "M03A", "vecuronium":       "M03A",
    "succinylcholine":  "M03A", "pancuronium":      "M03A", "atracurium":       "M03A",
    # N
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
    # P
    "hydroxychloroquine":"P01B",
    # R
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
    # V
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


ATC3_PATTERN = re.compile(r"\b([A-Z]\d{2}[A-Z])\b")


def extract_atc3_from_output(raw_output: str, valid_atc3_codes: Set[str],
                              strict: bool = False) -> List[str]:
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


def evaluate_predictions(results: List[Dict], label: str) -> Dict[str, float]:
    f1_scores, jaccard_scores = [], []
    n_total = n_nonempty_true = n_nonempty_pred = 0
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
        "method": label, "n_total": n_total,
        "n_with_true": n_nonempty_true, "n_with_pred": n_nonempty_pred,
        "f1_score": round(avg_f1, 4), "jaccard_similarity": round(avg_jaccard, 4),
    }


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

    drug_names_per_adm = (
        rx_mapped.groupby(["subject_id", "hadm_id"])["drug"]
        .apply(lambda x: list(x.unique())[:10])
        .reset_index()
        .rename(columns={"drug": "drug_names"})
    )

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
    adm["visit_num"] = adm.groupby("subject_id").cumcount() + 1
    adm["total_visits"] = adm.groupby("subject_id")["hadm_id"].transform("count")

    logger.info(f"  Total admissions: {len(adm):,}")
    logger.info(f"  Unique patients: {adm['subject_id'].nunique():,}")

    return adm


# equalised feature set vs the GRU baseline (demographics, full lists, time deltas, visit idx)
def build_enhanced_prompt(visits: pd.DataFrame, include_last_rx: bool = False,
                          max_history_visits: int = 8) -> str:
    total = len(visits)
    if total > max_history_visits + 1:
        visits = visits.iloc[-(max_history_visits + 1):]

    actual = len(visits)

    first = visits.iloc[0]
    gender = str(first.get("gender", "unknown"))
    age = first.get("anchor_age", 0)
    parts = [f"Patient: {gender}, age {age}. Total ICU visits: {total}."]

    prev_time = None
    ordinals = ["first", "second", "third", "fourth", "fifth",
                "sixth", "seventh", "eighth", "ninth", "tenth"]

    for i, (_, v) in enumerate(visits.iterrows()):
        is_last = (i == actual - 1)
        ord_str = ordinals[min(i, len(ordinals) - 1)] if i < len(ordinals) else f"{i+1}th"

        # days since previous admit
        curr_time = v.get("admittime", None)
        delta_str = ""
        if prev_time is not None and curr_time is not None:
            try:
                delta_days = (pd.Timestamp(curr_time) - pd.Timestamp(prev_time)).days
                delta_str = f" ({delta_days} days later)"
            except Exception:
                pass
        prev_time = curr_time

        diag_list = v["diagnoses"] if isinstance(v["diagnoses"], list) else []
        diag_str = ", ".join(diag_list) if diag_list else "unspecified"

        if is_last and not include_last_rx:
            parts.append(
                f"Visit {i+1}/{total}{delta_str}: diagnosis: {diag_str}. "
                f"Prescribe ATC-3 codes:"
            )
        else:
            atc3_list = v["atc3_codes"] if isinstance(v["atc3_codes"], list) else []
            rx_str = ", ".join(atc3_list) if atc3_list else "none"

            parts.append(
                f"Visit {i+1}/{total}{delta_str}: diagnosis: {diag_str}. "
                f"Prescribed: {rx_str}."
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


class MedPredDataset(Dataset):
    """Causal LM dataset. Prompt tokens masked to -100; only target contributes to loss."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids_list = []
        self.labels_list = []

        for ex in tqdm(examples, desc="Tokenizing training data"):
            prompt = ex["prompt"]
            target = ", ".join(sorted(ex["target_atc3"]))

            full_text = prompt + " " + target + tokenizer.eos_token

            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            input_ids = encoded["input_ids"]

            # mask prompt; loss only on target
            prompt_encoded = tokenizer(
                prompt + " ",
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            prompt_len = len(prompt_encoded["input_ids"])

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[:len(input_ids)]

            self.input_ids_list.append(input_ids)
            self.labels_list.append(labels)

        logger.info(f"  Tokenized {len(self.input_ids_list)} training examples")

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids_list[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels_list[idx], dtype=torch.long),
        }


class PaddingCollator:
    # left-pad to max len in batch (causal LM requirement)
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(torch.cat([torch.full((pad_len,), self.pad_id, dtype=torch.long),
                                        b["input_ids"]]))
            labels.append(torch.cat([torch.full((pad_len,), -100, dtype=torch.long),
                                     b["labels"]]))
            attention_mask.append(torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                             torch.ones(len(b["input_ids"]), dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


def build_rag_prompt(patient_prompt: str, similar_cases: List[str]) -> str:
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


def run_inference(prompts: List[str], tokenizer, model, device: str,
                  batch_size: int = BATCH_SIZE) -> List[str]:
    all_outputs = []
    model.eval()
    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="BioMistral-FT inference"):
        batch_prompts = prompts[start_idx:start_idx + batch_size]
        encoded = tokenizer(
            batch_prompts, padding=True, truncation=True,
            max_length=MAX_PROMPT_TOKENS, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=4,                    # bumped from 2 to match eICU run
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = encoded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
        all_outputs.extend(outputs)
    return all_outputs


def main():
    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected!")

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
    logger.info(f"  Eligible patients: {len(eligible_patients):,}")

    # 70/10/20, same split as GRU
    all_pids = sorted(eligible_patients.keys())
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_pids)

    split_idx = int(len(all_pids) * TRAIN_RATIO)
    train_val_pids = all_pids[:split_idx]
    test_pids = all_pids[split_idx:]

    val_split = int(len(train_val_pids) * (1 - VAL_RATIO / TRAIN_RATIO))
    train_pids = train_val_pids[:val_split]
    val_pids = train_val_pids[val_split:]

    logger.info(f"  Train: {len(train_pids):,}  Val: {len(val_pids):,}  Test: {len(test_pids):,}")

    all_atc3_set: Set[str] = set()
    for pid in all_pids:
        for _, row in eligible_patients[pid].iterrows():
            if isinstance(row["atc3_codes"], list):
                all_atc3_set.update(row["atc3_codes"])
    logger.info(f"  Unique ATC-3 codes: {len(all_atc3_set)}")

    t_preprocess = time.time()

    logger.info("=" * 70)
    logger.info("STEP 3: PREPARING FINE-TUNING DATA")
    logger.info("=" * 70)

    train_examples: List[Dict] = []
    val_examples: List[Dict] = []
    if SKIP_TRAIN:
        logger.info("SKIP_TRAIN=1 — skipping train/val example construction")
    else:
        for pid in tqdm(train_pids, desc="Building train examples"):
            group = eligible_patients[pid]
            prompt = build_enhanced_prompt(group, include_last_rx=False)
            last_visit = group.iloc[-1]
            target_atc3 = last_visit["atc3_codes"] if isinstance(last_visit["atc3_codes"], list) else []
            if target_atc3:
                train_examples.append({"prompt": prompt, "target_atc3": target_atc3})

        for pid in tqdm(val_pids, desc="Building val examples"):
            group = eligible_patients[pid]
            prompt = build_enhanced_prompt(group, include_last_rx=False)
            last_visit = group.iloc[-1]
            target_atc3 = last_visit["atc3_codes"] if isinstance(last_visit["atc3_codes"], list) else []
            if target_atc3:
                val_examples.append({"prompt": prompt, "target_atc3": target_atc3})

        logger.info(f"  Train examples: {len(train_examples):,}")
        logger.info(f"  Val examples:   {len(val_examples):,}")

    t_data_prep = time.time()

    logger.info("=" * 70)
    logger.info("STEP 4: LOADING BioMistral + LoRA")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    adapter_path = OUT_DIR / ADAPTER_DIR_NAME

    # 4-bit quant. IMPORTANT: bnb_4bit_compute_dtype MUST match trainer precision (bf16) —
    # mixing fp16 compute with bf16 trainer caused NaN gradient blowup last run.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if SKIP_TRAIN:
        # inference-only: base + LoRA, no training
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"SKIP_TRAIN=1 set but no adapter found at {adapter_path}. "
                "Run training first (unset SKIP_TRAIN) or point ADAPTER_DIR_NAME "
                "to a valid adapter directory."
            )
        logger.info(f"SKIP_TRAIN=1 — loading existing adapter from {adapter_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
        t_model_load = time.time()
        t_finetuning = t_model_load  # skipped training
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        t_model_load = time.time()

        logger.info("Building tokenized datasets...")
        train_dataset = MedPredDataset(train_examples, tokenizer, MAX_SEQ_LEN)
        val_dataset = MedPredDataset(val_examples, tokenizer, MAX_SEQ_LEN)

        logger.info("=" * 70)
        logger.info("STEP 5: FINE-TUNING BioMistral WITH LoRA")
        logger.info("=" * 70)

        training_args = TrainingArguments(
            output_dir=str(OUT_DIR / "checkpoints"),
            num_train_epochs=FT_EPOCHS,
            per_device_train_batch_size=FT_BATCH_SIZE,
            per_device_eval_batch_size=FT_BATCH_SIZE,
            gradient_accumulation_steps=FT_GRAD_ACCUM,
            learning_rate=FT_LR,
            warmup_ratio=FT_WARMUP_RATIO,
            weight_decay=0.01,
            max_grad_norm=FT_MAX_GRAD_NORM,   # NaN guard
            fp16=False,
            bf16=True,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=PaddingCollator(tokenizer),
        )

        logger.info("Starting fine-tuning...")
        trainer.train()
        logger.info("Fine-tuning complete!")

        t_finetuning = time.time()

        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logger.info(f"Saved LoRA adapter to {adapter_path}")

    # release optimizer/grad buffers before generate()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    logger.info("Building test patient prompts...")
    test_data = []
    for pid in tqdm(test_pids, desc="Test patient prompts"):
        group = eligible_patients[pid]
        last_visit = group.iloc[-1]
        true_atc3 = last_visit["atc3_codes"] if isinstance(last_visit["atc3_codes"], list) else []
        baseline_prompt = build_enhanced_prompt(group, include_last_rx=False)
        test_data.append({
            "pid": pid, "baseline_prompt": baseline_prompt,
            "true_atc3": true_atc3,
        })
    logger.info(f"  Test samples: {len(test_data):,}")

    logger.info("\n" + "=" * 70)
    logger.info("PHASE A: BASELINE Fine-tuned BioMistral (No RAG)")
    logger.info("=" * 70)

    baseline_prompts = [td["baseline_prompt"] for td in test_data]
    logger.info(f"\n--- Sample prompt ---")
    logger.info(baseline_prompts[0][:600])
    logger.info("---\n")

    baseline_outputs = run_inference(baseline_prompts, tokenizer, model, device, BATCH_SIZE)

    baseline_results = []
    for td, raw_output in zip(test_data, baseline_outputs):
        pred_atc3 = extract_atc3_from_output(raw_output, all_atc3_set, strict=False)
        baseline_results.append({
            "pid": td["pid"], "true_atc3": td["true_atc3"],
            "pred_atc3": pred_atc3, "raw_output": raw_output[:300],
        })

    for i in range(min(5, len(baseline_results))):
        r = baseline_results[i]
        logger.info(f"  Patient {r['pid']}: true={r['true_atc3'][:5]} "
                     f"pred={r['pred_atc3'][:5]} raw='{r['raw_output'][:100]}'")

    t_inference = time.time()

    baseline_metrics = evaluate_predictions(
        baseline_results, "Fine-tuned BioMistral (No RAG)")

    baseline_df = pd.DataFrame([{
        "subject_id": r["pid"], "true_atc3": json.dumps(r["true_atc3"]),
        "pred_atc3": json.dumps(r["pred_atc3"]), "raw_output": r["raw_output"],
    } for r in baseline_results])
    baseline_df.to_csv(OUT_DIR / "baseline_results.csv", index=False)

    t_evaluation = time.time()

    logger.info(f"\n{'='*70}")
    logger.info("TIME COST BREAKDOWN")
    logger.info(f"{'='*70}")
    logger.info(f"  Data preprocessing:      {t_preprocess - t_start:.1f}s")
    logger.info(f"  Data preparation:        {t_data_prep - t_preprocess:.1f}s")
    logger.info(f"  Model loading:           {t_model_load - t_data_prep:.1f}s")
    logger.info(f"  Fine-tuning:             {t_finetuning - t_model_load:.1f}s")
    logger.info(f"  Inference:               {t_inference - t_finetuning:.1f}s")
    logger.info(f"  Evaluation:              {t_evaluation - t_inference:.1f}s")
    logger.info(f"  Total:                   {t_evaluation - t_start:.1f}s ({(t_evaluation - t_start)/60:.1f} min)")

    summary = {
        "finetuned_metrics": baseline_metrics,
        "dataset_stats": {
            "total_patients": len(all_pids),
            "train_patients": len(train_pids),
            "val_patients": len(val_pids),
            "test_patients": len(test_pids),
            "unique_atc3_codes": len(all_atc3_set),
        },
        "config": {
            "model": "BioMistral-7B",
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "ft_epochs": FT_EPOCHS,
            "ft_lr": FT_LR,
            "random_state": RANDOM_STATE,
        },
        "time_cost": {
            "data_preprocessing_sec": round(t_preprocess - t_start, 2),
            "data_preparation_sec": round(t_data_prep - t_preprocess, 2),
            "model_loading_sec": round(t_model_load - t_data_prep, 2),
            "finetuning_sec": round(t_finetuning - t_model_load, 2),
            "inference_sec": round(t_inference - t_finetuning, 2),
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
