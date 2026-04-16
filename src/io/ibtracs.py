from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests


IBTRACS_VERSION = "v04r01"
IBTRACS_BASE_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
)

def download_ibtracs_csv(
    output_path: str | Path,
    subset: str = "since1980",
    timeout: int = 120,
    chunk_size: int = 1024 * 1024,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset = subset.strip().lower()
    if subset == "since1980":
        filename = f"ibtracs.since1980.list.{IBTRACS_VERSION}.csv"
    elif subset == "all":
        filename = f"ibtracs.ALL.list.{IBTRACS_VERSION}.csv"
    else:
        raise ValueError(f"Unsupported subset: {subset!r}")

    url = f"{IBTRACS_BASE_URL}/{IBTRACS_VERSION}/access/csv/{filename}"
    print("Downloading:", url)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    return output_path


def read_ibtracs_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    df = pd.read_csv(
        path,
        compression="gzip" if path.suffix == ".gz" else None,
        dtype=str,
        low_memory=False,
        na_values=["", " ", "NA", "NOT_NAMED", "NOT_NAMED ", "nan"],
        keep_default_na=True,
    )

    if "ISO_TIME" in df.columns:
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")

    return df


def _csv_filename(subset: str) -> str:
    subset_to_name = {
        "since1980": f"ibtracs.since1980.list.{IBTRACS_VERSION}.csv",
        "all": f"ibtracs.ALL.list.{IBTRACS_VERSION}.csv",
        "active": f"ibtracs.ACTIVE.list.{IBTRACS_VERSION}.csv",
        "last3years": f"ibtracs.last3years.list.{IBTRACS_VERSION}.csv",
        "na": f"ibtracs.NA.list.{IBTRACS_VERSION}.csv",
        "ep": f"ibtracs.EP.list.{IBTRACS_VERSION}.csv",
        "wp": f"ibtracs.WP.list.{IBTRACS_VERSION}.csv",
        "ni": f"ibtracs.NI.list.{IBTRACS_VERSION}.csv",
        "si": f"ibtracs.SI.list.{IBTRACS_VERSION}.csv",
        "sp": f"ibtracs.SP.list.{IBTRACS_VERSION}.csv",
        "sa": f"ibtracs.SA.list.{IBTRACS_VERSION}.csv",
    }
    try:
        return subset_to_name[subset]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported subset {subset!r}. Valid options: {sorted(subset_to_name)}"
        ) from exc


def _normalize_subset(subset: str) -> str:
    return subset.strip().lower()


def clean_ibtracs(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    data = df.copy()

    required_cols = ["SID", "ISO_TIME", "LAT", "LON"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"IBTrACS dataframe is missing required columns: {missing}")

    data = _drop_header_repeat_rows(data)

    data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"], utc=True, errors="coerce")

    numeric_cols = [
        "LAT",
        "LON",
        "WMO_WIND",
        "WMO_PRES",
        "USA_WIND",
        "USA_PRES",
        "TOKYO_WIND",
        "TOKYO_PRES",
        "CMA_WIND",
        "CMA_PRES",
        "HKO_WIND",
        "HKO_PRES",
        "NEWDELHI_WIND",
        "NEWDELHI_PRES",
        "REUNION_WIND",
        "REUNION_PRES",
        "BOM_WIND",
        "BOM_PRES",
        "NADI_WIND",
        "NADI_PRES",
        "WELLINGTON_WIND",
        "WELLINGTON_PRES",
        "MLC_WIND",
        "MLC_PRES",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"]).copy()

    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC")
        data = data.loc[data["ISO_TIME"] >= start_ts].copy()

    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC")
        data = data.loc[data["ISO_TIME"] <= end_ts].copy()

    data["lon_360"] = data["LON"] % 360.0
    data["lon_180"] = ((data["LON"] + 180.0) % 360.0) - 180.0

    wind_candidates = [col for col in ["USA_WIND", "WMO_WIND", "TOKYO_WIND", "CMA_WIND",
                                       "HKO_WIND", "NEWDELHI_WIND", "REUNION_WIND",
                                       "BOM_WIND", "NADI_WIND", "WELLINGTON_WIND", "MLC_WIND"]
                       if col in data.columns]
    pres_candidates = [col for col in ["USA_PRES", "WMO_PRES", "TOKYO_PRES", "CMA_PRES",
                                       "HKO_PRES", "NEWDELHI_PRES", "REUNION_PRES",
                                       "BOM_PRES", "NADI_PRES", "WELLINGTON_PRES", "MLC_PRES"]
                       if col in data.columns]

    data["wind_kt"] = _coalesce_columns(data, wind_candidates)
    data["pres_mb"] = _coalesce_columns(data, pres_candidates)

    rename_map = {}
    if "BASIN" in data.columns:
        rename_map["BASIN"] = "BASIN"
    if "NAME" in data.columns:
        rename_map["NAME"] = "NAME"
    if "SEASON" in data.columns:
        rename_map["SEASON"] = "SEASON"
    if "NATURE" in data.columns:
        rename_map["NATURE"] = "NATURE"
    if "TRACK_TYPE" in data.columns:
        rename_map["TRACK_TYPE"] = "TRACK_TYPE"

    data = data.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    keep_first = [
        "SID",
        "SEASON",
        "BASIN",
        "NAME",
        "ISO_TIME",
        "LAT",
        "LON",
        "lon_180",
        "lon_360",
        "wind_kt",
        "pres_mb",
        "NATURE",
        "TRACK_TYPE",
    ]
    remaining = [c for c in data.columns if c not in keep_first]
    ordered_cols = [c for c in keep_first if c in data.columns] + remaining
    data = data[ordered_cols]

    return data


def build_storm_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ["SID", "ISO_TIME", "LAT", "lon_180"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Point dataframe is missing required columns: {missing}")

    working = df.copy().sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    group = working.groupby("SID", sort=False)

    summary = group.agg(
        start_time=("ISO_TIME", "min"),
        end_time=("ISO_TIME", "max"),
        n_points=("ISO_TIME", "size"),
        season=("SEASON", _first_valid) if "SEASON" in working.columns else ("ISO_TIME", "size"),
        basin=("BASIN", _first_valid) if "BASIN" in working.columns else ("ISO_TIME", "size"),
        name=("NAME", _first_valid) if "NAME" in working.columns else ("ISO_TIME", "size"),
        nature=("NATURE", _first_valid) if "NATURE" in working.columns else ("ISO_TIME", "size"),
        track_type=("TRACK_TYPE", _first_valid) if "TRACK_TYPE" in working.columns else ("ISO_TIME", "size"),
        min_lat=("LAT", "min"),
        max_lat=("LAT", "max"),
        min_lon_180=("lon_180", "min"),
        max_lon_180=("lon_180", "max"),
        max_wind_kt=("wind_kt", "max") if "wind_kt" in working.columns else ("ISO_TIME", "size"),
        min_pres_mb=("pres_mb", "min") if "pres_mb" in working.columns else ("ISO_TIME", "size"),
    ).reset_index()

    summary["duration_hours"] = (
        (summary["end_time"] - summary["start_time"]).dt.total_seconds() / 3600.0
    )

    if "max_wind_kt" in summary.columns:
        summary["has_wind"] = summary["max_wind_kt"].notna()
    if "min_pres_mb" in summary.columns:
        summary["has_pressure"] = summary["min_pres_mb"].notna()

    summary = summary.sort_values(["start_time", "SID"]).reset_index(drop=True)
    return summary


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    return path


def _drop_header_repeat_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "SID" not in df.columns:
        return df
    mask = df["SID"].astype(str).str.upper().eq("SID")
    if "ISO_TIME" in df.columns:
        mask = mask | df["ISO_TIME"].astype(str).str.upper().eq("ISO_TIME")
    return df.loc[~mask].copy()


def _coalesce_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")

    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    for col in cols:
        if col in df.columns:
            out = out.fillna(df[col].astype("Float64"))
    return out


def _first_valid(series: pd.Series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return pd.NA
    return non_null.iloc[0]
