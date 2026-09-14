"""Build and clean the ED master dataset used by the notebook.

Expected raw MIMIC-IV-ED-derived CSV files:
- edstays.csv
- triage.csv
- vitalsign.csv
- diagnosis.csv
- medrecon.csv

Each file may either be directly inside --raw-dir or inside a subfolder with
that same filename, mirroring the layouts used in the original Colab notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .config import ALL_NUMERIC_CANDIDATES, CLIP_RANGES, PROCESSED_DIR, RAW_DIR


def resolve_csv(raw_dir: Path, filename: str) -> Path:
    direct = raw_dir / filename
    nested = raw_dir / filename / filename
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Cannot find {filename} in {raw_dir} (direct or nested layout).")


def build_master(raw_dir: Path) -> pd.DataFrame:
    edstays = pd.read_csv(resolve_csv(raw_dir, "edstays.csv"))
    triage = pd.read_csv(resolve_csv(raw_dir, "triage.csv"))
    vitals = pd.read_csv(resolve_csv(raw_dir, "vitalsign.csv"))
    diagnosis = pd.read_csv(resolve_csv(raw_dir, "diagnosis.csv"))
    medrecon = pd.read_csv(resolve_csv(raw_dir, "medrecon.csv"))

    triage_agg = triage.sort_values("stay_id").groupby("stay_id", as_index=False).first()
    master = edstays.merge(triage_agg, on="stay_id", how="left", suffixes=("_ed", "_triage"))

    vitals["charttime"] = pd.to_datetime(vitals["charttime"], errors="coerce")
    base_vitals = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
    for col in base_vitals:
        if col in vitals.columns:
            vitals[col] = pd.to_numeric(vitals[col], errors="coerce")

    vitals = vitals.sort_values(["stay_id", "charttime"])
    available = [c for c in base_vitals if c in vitals.columns]
    agg_dict = {c: ["first", "last", "mean", "min", "max"] for c in available}
    agg_dict["charttime"] = ["count"]
    vitals_agg = vitals.groupby("stay_id").agg(agg_dict)
    vitals_agg.columns = ["_".join(col).strip() for col in vitals_agg.columns]
    vitals_agg = vitals_agg.reset_index().rename(columns={"charttime_count": "vitals_count"})
    master = master.merge(vitals_agg, on="stay_id", how="left")

    if "icd_code" in diagnosis.columns:
        diag_agg = (
            diagnosis.groupby("stay_id")["icd_code"]
            .apply(lambda x: ";".join(x.astype(str)))
            .reset_index(name="diagnosis_codes")
        )
        master = master.merge(diag_agg, on="stay_id", how="left")

    if "name" in medrecon.columns:
        med_agg = (
            medrecon.groupby("stay_id")["name"]
            .apply(lambda x: ";".join(x.astype(str)))
            .reset_index(name="medications")
        )
        master = master.merge(med_agg, on="stay_id", how="left")

    return master


def clean_master(df: pd.DataFrame) -> pd.DataFrame:
    if "acuity" not in df.columns:
        raise KeyError("The master dataset must contain an 'acuity' column.")

    out = df.copy()
    out["_acuity_num"] = pd.to_numeric(out["acuity"], errors="coerce")
    out = out.loc[out["_acuity_num"].between(1, 5, inclusive="both")].copy()
    out["acuity"] = out["_acuity_num"].astype(int)
    out.drop(columns="_acuity_num", inplace=True)

    numeric_cols = [c for c in ALL_NUMERIC_CANDIDATES if c in out.columns]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")

    for col, (lo, hi) in CLIP_RANGES.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and clean the ESI master dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--master-out", type=Path, default=PROCESSED_DIR / "master_dataset.csv")
    parser.add_argument("--clean-out", type=Path, default=PROCESSED_DIR / "master_dataset_clean.csv")
    args = parser.parse_args()

    args.master_out.parent.mkdir(parents=True, exist_ok=True)
    args.clean_out.parent.mkdir(parents=True, exist_ok=True)

    master = build_master(args.raw_dir)
    master.to_csv(args.master_out, index=False)
    clean = clean_master(master)
    clean.to_csv(args.clean_out, index=False)

    print(f"Master dataset: {master.shape} -> {args.master_out}")
    print(f"Clean dataset:  {clean.shape} -> {args.clean_out}")
    print("Acuity distribution:")
    print(clean["acuity"].value_counts().sort_index())


if __name__ == "__main__":
    main()
