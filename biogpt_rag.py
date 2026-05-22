#!/usr/bin/env python3
"""BioGPT with BioLORD-2023 retrieval; per-stage timings logged."""


from __future__ import annotations
import os
import re
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm
import requests

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
OUT_DIR = Path("/workspace/LLM_research/treatRag/output_biogpt_rag")
CHROMA_DB_DIR = OUT_DIR / "chroma_db"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_RATIO = 0.8
MAX_PROMPT_TOKENS = 850    # BioGPT ctx 1024 minus generation room
MAX_NEW_TOKENS = 128
BIOGPT_MODEL = "microsoft/biogpt"
BATCH_SIZE = 64

EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_NORMALIZE = True
CHROMA_COLLECTION_NAME = "treatrag_biolord"
CHROMA_DISTANCE_METRIC = "cosine"

RAG_TOP_K = 30             # pull this many from chroma, then filter
RAG_MAX_CASES = 10         # paper caps at <10
RAG_MIN_SIMILARITY = 0.3
RAG_THRESHOLD_MULT = 1.5   # adaptive gap multiplier

# NDC -> RXCUI -> ATC-3 via RxNorm (Liu et al.)
NDC_ATC3_CACHE_PATH = CSV_DIR / "ndc_to_atc3_cache.json"
DRUGNAME_ATC3_CACHE_PATH = CSV_DIR / "drugname_to_atc3_cache.json"
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXNORM_MAX_WORKERS = 8
RXNORM_TIMEOUT = 10


def _ndc_to_rxcui(ndc: str) -> Optional[str]:
    try:
        url = f"{RXNORM_BASE_URL}/ndcstatus.json?ndc={ndc}"
        resp = requests.get(url, timeout=RXNORM_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            rxcui = data.get("ndcStatus", {}).get("rxcui")
            if rxcui and rxcui != "0":
                return rxcui
    except Exception:
        pass
    return None


def _drugname_to_rxcui(drug_name: str) -> Optional[str]:
    # approximate match — exact term often fails on MIMIC drug strings
    try:
        url = f"{RXNORM_BASE_URL}/approximateTerm.json?term={requests.utils.quote(drug_name)}&maxEntries=1"
        resp = requests.get(url, timeout=RXNORM_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("approximateGroup", {}).get("candidate", [])
            if candidates:
                return candidates[0].get("rxcui")
    except Exception:
        pass
    return None


def _rxcui_to_atc3(rxcui: str) -> Optional[str]:
    try:
        url = f"{RXNORM_BASE_URL}/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC"
        resp = requests.get(url, timeout=RXNORM_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            infos = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            for info in infos:
                class_id = info.get("rxclassMinConceptItem", {}).get("classId", "")
                class_type = info.get("rxclassMinConceptItem", {}).get("classType", "")
                # ATC1-4 = level 4 (5 chars). truncate to 4 chars for level 3
                if class_type == "ATC1-4" and len(class_id) >= 4:
                    return class_id[:4]
            # fallback when class_type isn't reported
            for info in infos:
                class_id = info.get("rxclassMinConceptItem", {}).get("classId", "")
                if len(class_id) >= 4:
                    return class_id[:4]
    except Exception:
        pass
    return None


def _resolve_single_ndc(ndc: str) -> Tuple[str, Optional[str]]:
    rxcui = _ndc_to_rxcui(ndc)
    if rxcui:
        atc3 = _rxcui_to_atc3(rxcui)
        if atc3:
            return ndc, atc3
    return ndc, None


def _resolve_single_drugname(drug_name: str) -> Tuple[str, Optional[str]]:
    rxcui = _drugname_to_rxcui(drug_name)
    if rxcui:
        atc3 = _rxcui_to_atc3(rxcui)
        if atc3:
            return drug_name, atc3
    return drug_name, None


def build_ndc_to_atc3_mapping(ndc_series: pd.Series) -> Dict[str, str]:
    """NDC -> ATC-3 via RxNorm. Disk-cached so re-runs are cheap."""
    cache: Dict[str, str] = {}
    if NDC_ATC3_CACHE_PATH.exists():
        with open(NDC_ATC3_CACHE_PATH) as f:
            cache = json.load(f)
        logger.info(f"  Loaded NDC->ATC3 cache: {len(cache):,} entries")

    unique_ndcs = set(ndc_series.dropna().astype(str).str.strip())
    unique_ndcs.discard("")
    unique_ndcs.discard("0")
    uncached = [ndc for ndc in unique_ndcs if ndc not in cache]

    if uncached:
        logger.info(f"  Resolving {len(uncached):,} uncached NDCs via RxNorm API...")
        resolved = 0
        with ThreadPoolExecutor(max_workers=RXNORM_MAX_WORKERS) as pool:
            futures = {pool.submit(_resolve_single_ndc, ndc): ndc for ndc in uncached}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="NDC->ATC3 API"):
                ndc, atc3 = future.result()
                if atc3:
                    cache[ndc] = atc3
                    resolved += 1
                else:
                    cache[ndc] = ""  # remember we tried
        logger.info(f"  Resolved {resolved:,}/{len(uncached):,} NDCs to ATC-3")

        with open(NDC_ATC3_CACHE_PATH, "w") as f:
            json.dump(cache, f)
        logger.info(f"  Saved NDC->ATC3 cache -> {NDC_ATC3_CACHE_PATH}")

    return {ndc: atc3 for ndc, atc3 in cache.items() if atc3}


def build_drugname_to_atc3_mapping(drug_series: pd.Series) -> Dict[str, str]:
    """Fallback drug-name mapping for entries with no usable NDC."""
    cache: Dict[str, str] = {}
    if DRUGNAME_ATC3_CACHE_PATH.exists():
        with open(DRUGNAME_ATC3_CACHE_PATH) as f:
            cache = json.load(f)
        logger.info(f"  Loaded drug name->ATC3 cache: {len(cache):,} entries")

    unique_drugs = set(drug_series.dropna().astype(str).str.strip().str.lower())
    unique_drugs.discard("")
    uncached = [d for d in unique_drugs if d not in cache]

    if uncached:
        logger.info(f"  Resolving {len(uncached):,} uncached drug names via RxNorm API...")
        resolved = 0
        with ThreadPoolExecutor(max_workers=RXNORM_MAX_WORKERS) as pool:
            futures = {pool.submit(_resolve_single_drugname, d): d for d in uncached}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="DrugName->ATC3 API"):
                drug, atc3 = future.result()
                if atc3:
                    cache[drug] = atc3
                    resolved += 1
                else:
                    cache[drug] = ""
        logger.info(f"  Resolved {resolved:,}/{len(uncached):,} drug names to ATC-3")

        with open(DRUGNAME_ATC3_CACHE_PATH, "w") as f:
            json.dump(cache, f)
        logger.info(f"  Saved drug name->ATC3 cache -> {DRUGNAME_ATC3_CACHE_PATH}")

    return {d: atc3 for d, atc3 in cache.items() if atc3}


def normalize_drug_name(name: str) -> str:
    # strip dose, formulation, parens, collapse whitespace
    name = str(name).lower().strip()
    name = re.sub(r"\d+\.?\d*\s*(mg|mcg|ml|g|units?|%|meq)\b.*", "", name)
    name = re.sub(r"\s*(tablet|capsule|injection|solution|suspension|cream|"
                  r"ointment|syrup|patch|suppository|inhaler|vial|bag|"
                  r"powder|liquid|drops|spray|gel|lotion|iv|oral|topical|"
                  r"ophthalmic|otic|nasal|rectal|sublingual|transdermal)\b.*", "", name)
    name = re.sub(r"\s*\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_and_preprocess_data() -> pd.DataFrame:
    """Join MIMIC tables, map drugs to ATC-3, one row per admission."""
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

    logger.info(f"  Patients:      {len(patients):,}")
    logger.info(f"  Admissions:    {len(admissions):,}")
    logger.info(f"  Prescriptions: {len(prescriptions):,}")
    logger.info(f"  DRG codes:     {len(drgcodes):,}")

    # diagnoses come from DRG; prefer APR if present, else all rows
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

    logger.info("Mapping drugs to ATC-3 codes via RxNorm (NDC -> RXCUI -> ATC-3)...")
    prescriptions["drug"] = prescriptions["drug"].fillna("").astype(str)
    # NDC: pandas reads numeric and drops leading zeros — re-pad to 11 digits
    prescriptions["ndc"] = (
        prescriptions["ndc"].fillna("").astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .apply(lambda x: x.zfill(11) if x and x != "0" and x.isdigit() else x)
    )

    ndc_to_atc3 = build_ndc_to_atc3_mapping(prescriptions["ndc"])
    logger.info(f"  NDC->ATC3 mapping entries: {len(ndc_to_atc3):,}")

    drugname_to_atc3 = build_drugname_to_atc3_mapping(prescriptions["drug"])
    logger.info(f"  DrugName->ATC3 mapping entries: {len(drugname_to_atc3):,}")

    # vectorised: NDC first, fall back to drug name
    prescriptions["atc3_ndc"] = prescriptions["ndc"].map(ndc_to_atc3)

    prescriptions["drug_norm"] = prescriptions["drug"].str.lower().str.strip()
    prescriptions["drug_norm"] = prescriptions["drug_norm"].str.replace(
        r"\d+\.?\d*\s*(mg|mcg|ml|g|units?|%|meq)\b.*", "", regex=True)
    prescriptions["drug_norm"] = prescriptions["drug_norm"].str.replace(
        r"\s*(tablet|capsule|injection|solution|suspension|cream|"
        r"ointment|syrup|patch|suppository|inhaler|vial|bag|"
        r"powder|liquid|drops|spray|gel|lotion|iv|oral|topical|"
        r"ophthalmic|otic|nasal|rectal|sublingual|transdermal)\b.*", "", regex=True)
    prescriptions["drug_norm"] = prescriptions["drug_norm"].str.replace(
        r"\s*\(.*?\)", "", regex=True).str.strip()
    prescriptions["atc3_drug"] = prescriptions["drug_norm"].map(drugname_to_atc3)

    prescriptions["atc3"] = prescriptions["atc3_ndc"].fillna(prescriptions["atc3_drug"])
    prescriptions.drop(columns=["atc3_ndc", "atc3_drug", "drug_norm"], inplace=True)

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

    # stash drug->atc3 for output parsing later
    global _RUNTIME_DRUGNAME_ATC3
    _RUNTIME_DRUGNAME_ATC3 = dict(drugname_to_atc3)
    logger.info(f"  Runtime drug name->ATC3 lookup: {len(_RUNTIME_DRUGNAME_ATC3):,} entries")

    return adm


def build_patient_prompt(visits: pd.DataFrame, include_last_rx: bool = False,
                         max_history_visits: int = 4,
                         use_drug_names: bool = False) -> str:
    """Figure-2 style prompt: history visits + last visit diagnosis."""
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
    """BioLORD-2023 + ChromaDB dense retrieval. Replaces the n-gram Jaccard variant."""

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
        """Dense retrieval; filters out test PIDs to avoid leakage."""
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
                n_results=n_results * 2,  # over-fetch; we'll drop test pids
                include=["metadatas", "distances"],
            )

            for i in range(len(batch_embs)):
                retrieved_cases = []
                for md, dist in zip(results["metadatas"][i], results["distances"][i]):
                    retrieved_pid = int(md["pid"])
                    if retrieved_pid in query_pid_set:
                        continue
                    sim = 1.0 - float(dist)  # cosine dist -> sim
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
    """Cut cases off at the first big gap (mean + k*std) — paper §4.2."""
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


def build_biogpt_prompt_no_rag(patient_prompt: str) -> str:
    return patient_prompt


def build_biogpt_prompt_with_rag(
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


# filled in by load_and_preprocess_data()
_RUNTIME_DRUGNAME_ATC3: Dict[str, str] = {}


def extract_atc3_from_output(
    raw_output: str,
    valid_atc3_codes: Set[str],
    strict: bool = False,
) -> List[str]:
    """Pull ATC-3 codes from generated text. strict mode = baseline (explicit only)."""
    if not raw_output:
        return []

    text = raw_output.strip()
    found_codes: Set[str] = set()

    # explicit ATC-3 hits
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
            if normalized in _RUNTIME_DRUGNAME_ATC3:
                found_codes.add(_RUNTIME_DRUGNAME_ATC3[normalized])
    else:
        text_lower = text.lower()
        for drug_name, atc3 in _RUNTIME_DRUGNAME_ATC3.items():
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
                normalized = normalize_drug_name(item)
                if normalized in _RUNTIME_DRUGNAME_ATC3:
                    found_codes.add(_RUNTIME_DRUGNAME_ATC3[normalized])

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


def run_biogpt_inference(
    prompts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> List[str]:
    all_outputs = []

    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="BioGPT inference"):
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

    n_excluded = {"<2_visits": 0, "no_diag": 0, "no_rx_last": 0, "no_rx_prior": 0}
    for pid, group in patient_groups.items():
        group = group.sort_values("admittime").reset_index(drop=True)
        if len(group) < 2:
            n_excluded["<2_visits"] += 1
            continue
        last_visit = group.iloc[-1]
        if not last_visit["diagnoses"] or len(last_visit["diagnoses"]) == 0:
            n_excluded["no_diag"] += 1
            continue
        if not last_visit["atc3_codes"] or len(last_visit["atc3_codes"]) == 0:
            n_excluded["no_rx_last"] += 1
            continue
        prior = group.iloc[:-1]
        if not any(len(codes) > 0 for codes in prior["atc3_codes"]):
            n_excluded["no_rx_prior"] += 1
            continue
        eligible_patients[pid] = group

    logger.info(f"  Excluded: {n_excluded}")
    logger.info(f"  Eligible patients (>=2 visits with data): {len(eligible_patients):,}")
    logger.info(f"  *** SAMPLE SIZE: {len(eligible_patients):,} patients ***")

    # patient-level split (no leakage across visits)
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

        # retrieval query = history visits + final visit's diagnoses
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
    logger.info("LOADING BioGPT MODEL")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BIOGPT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BIOGPT_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    logger.info(f"  Model loaded on {device}")

    t_model_load = time.time()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE B: TreatRAG BioGPT (With BioLORD-2023 Embedding RAG)")
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

        # top-N then trim via adaptive gap
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

            prompt = build_biogpt_prompt_with_rag(td["rag_prompt"], case_texts)
        else:
            prompt = build_biogpt_prompt_no_rag(td["baseline_prompt"])

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

    rag_outputs = run_biogpt_inference(
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

    rag_metrics = evaluate_predictions(rag_results, "TreatRAG BioGPT (BioLORD-2023 RAG)")

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
            "model": "BioGPT",
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
