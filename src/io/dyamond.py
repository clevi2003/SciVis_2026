from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DYAMOND_BASE_URL = "https://maritime.sealstorage.io/api/v0/s3/utah/ncsa_OpenVisus_dataset"


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    variable: str
    url: str


_DATASET_SPECS = {
    "atmos": {
        "U": DatasetSpec("atmos", "U", f"{DYAMOND_BASE_URL}/DYAMONDv2_atmosphere_U/IDX"),
        "V": DatasetSpec("atmos", "V", f"{DYAMOND_BASE_URL}/DYAMONDv2_atmosphere_V/IDX"),
        "P": DatasetSpec("atmos", "P", f"{DYAMOND_BASE_URL}/DYAMONDv2_atmosphere_P/IDX"),
        "T": DatasetSpec("atmos", "T", f"{DYAMOND_BASE_URL}/DYAMONDv2_atmosphere_T/IDX"),
        "W": DatasetSpec("atmos", "W", f"{DYAMOND_BASE_URL}/DYAMONDv2_atmosphere_W/IDX"),
    },
    "ocean": {
        "Theta": DatasetSpec("ocean", "Theta", f"{DYAMOND_BASE_URL}/DYAMONDv2_ocean_Theta/IDX"),
        "u": DatasetSpec("ocean", "u", f"{DYAMOND_BASE_URL}/DYAMONDv2_ocean_u/IDX"),
        "v": DatasetSpec("ocean", "v", f"{DYAMOND_BASE_URL}/DYAMONDv2_ocean_v/IDX"),
        "salt": DatasetSpec("ocean", "salt", f"{DYAMOND_BASE_URL}/DYAMONDv2_ocean_salt/IDX"),
    },
}


def available_variables(domain: str) -> list[str]:
    domain_key = _normalize_domain(domain)
    return list(_DATASET_SPECS[domain_key].keys())


def get_dataset_spec(domain: str, variable: str) -> DatasetSpec:
    domain_key = _normalize_domain(domain)
    try:
        return _DATASET_SPECS[domain_key][variable]
    except KeyError as exc:
        valid = ", ".join(available_variables(domain_key))
        raise ValueError(
            f"Unknown variable {variable!r} for domain {domain_key!r}. "
            f"Valid variables: {valid}"
        ) from exc


def open_dataset(domain: str, variable: str):
    spec = get_dataset_spec(domain, variable)
    try:
        import OpenVisus as ov
    except Exception as exc:
        raise ImportError(
            "OpenVisus could not be imported. Install it in the notebook environment "
            "before calling open_dataset."
        ) from exc

    try:
        db = ov.LoadDataset(spec.url)
    except Exception as exc:
        raise RuntimeError(f"Failed to open OpenVisus dataset at {spec.url}") from exc

    if db is None:
        raise RuntimeError(f"OpenVisus returned None for dataset {spec.url}")

    return db


def dataset_summary(db: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "url": _safe_call(db, "getUrl"),
        "logic_box": _coerce_box(_safe_call(db, "getLogicBox")),
        "max_resolution": _safe_call(db, "getMaxResolution"),
        "bits_per_block": _safe_call(db, "getBitsPerBlock"),
        "timesteps": _safe_call(db, "getTimesteps"),
        "field": None,
    }

    field = _safe_call(db, "getField")
    if field is not None:
        summary["field"] = {
            "name": _safe_call(field, "name"),
            "dtype": _safe_call(field, "dtype"),
        }

    logic_box = summary["logic_box"]
    if logic_box is not None:
        p1, p2 = logic_box
        shape = tuple(int(b - a) for a, b in zip(p1, p2))
        summary["shape"] = shape
        summary["dims"] = len(shape)
    else:
        summary["shape"] = None
        summary["dims"] = None

    return summary


def read_logic_box(
    db: Any,
    x: tuple[int, int],
    y: tuple[int, int],
    z: tuple[int, int] | None = None,
    time: tuple[int, int] | None = None,
    quality: int | None = None,
):
    try:
        import OpenVisus as ov
    except Exception as exc:
        raise ImportError(
            "OpenVisus could not be imported. Install it in the notebook environment "
            "before calling read_logic_box."
        ) from exc

    logic_box = _safe_call(db, "getLogicBox")
    if logic_box is None:
        raise RuntimeError("Dataset does not expose a logic box.")

    p1, p2 = _coerce_box(logic_box)
    dims = len(p1)

    if dims not in (3, 4):
        raise ValueError(f"Unsupported dataset dimensionality: {dims}")

    if dims == 3:
        if z is None:
            z = (p1[2], min(p1[2] + 1, p2[2]))
        box_p1 = [x[0], y[0], z[0]]
        box_p2 = [x[1], y[1], z[1]]
    else:
        if z is None:
            z = (p1[2], min(p1[2] + 1, p2[2]))
        if time is None:
            time = (p1[3], min(p1[3] + 1, p2[3]))
        box_p1 = [x[0], y[0], z[0], time[0]]
        box_p2 = [x[1], y[1], z[1], time[1]]

    box_p1 = [max(a, lo) for a, lo in zip(box_p1, p1)]
    box_p2 = [min(b, hi) for b, hi in zip(box_p2, p2)]

    if any(a >= b for a, b in zip(box_p1, box_p2)):
        raise ValueError(
            f"Requested logic box is empty after clipping. "
            f"Requested p1={box_p1}, p2={box_p2}, dataset bounds={(p1, p2)}"
        )

    access = db.createAccess()
    query = ov.BoxQuery(db, ov.BoxNi(box_p1, box_p2))
    if quality is not None:
        query.end_resolutions.push_back(int(quality))

    ok = db.executeBoxQuery(access, query)
    if not ok:
        raise RuntimeError("OpenVisus executeBoxQuery returned failure.")

    data = query.buffer
    if data is None:
        raise RuntimeError("OpenVisus query returned no buffer.")

    if hasattr(data, "toNumPy"):
        return data.toNumPy()
    if hasattr(data, "c_ptr"):
        raise RuntimeError("Received an OpenVisus buffer that could not be converted to NumPy.")
    return data


def _normalize_domain(domain: str) -> str:
    domain_key = domain.strip().lower()
    if domain_key not in _DATASET_SPECS:
        valid = ", ".join(_DATASET_SPECS.keys())
        raise ValueError(f"Unknown domain {domain!r}. Valid domains: {valid}")
    return domain_key


def _safe_call(obj: Any, name: str):
    if obj is None or not hasattr(obj, name):
        return None
    attr = getattr(obj, name)
    try:
        return attr() if callable(attr) else attr
    except Exception:
        return None


def _coerce_box(box: Any):
    if box is None:
        return None
    if isinstance(box, tuple) and len(box) == 2:
        return tuple(box[0]), tuple(box[1])
    if hasattr(box, "p1") and hasattr(box, "p2"):
        return tuple(box.p1), tuple(box.p2)
    try:
        seq = list(box)
        if len(seq) == 2:
            return tuple(seq[0]), tuple(seq[1])
    except Exception:
        pass
    raise TypeError(f"Could not coerce logic box of type {type(box)}")
