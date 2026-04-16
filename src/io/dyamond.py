from __future__ import annotations

from dataclasses import dataclass


DYAMOND_BASE_URL = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"


@dataclass(frozen=True)
class DatasetSpec:
    variable: str
    domain: str
    long_name: str
    idx_path: str
    has_depth: bool = True

    @property
    def url(self):
        return DYAMOND_BASE_URL + self.idx_path


_DATASET_SPECS = {
    "U": DatasetSpec("U", "atmosphere", "eastward wind", "https://maritime.sealstorage.io/api/visus/datasets/dyamond/U/visus.idx"),
    "V": DatasetSpec("V", "atmosphere", "northward wind", "https://maritime.sealstorage.io/api/visus/datasets/dyamond/V/visus.idx"),
    "P": DatasetSpec( "P", "atmosphere", "mid-level pressure", "https://maritime.sealstorage.io/api/visus/datasets/dyamond/P/visus.idx"),
    "T": DatasetSpec("T", "atmosphere", "air temperature", "https://maritime.sealstorage.io/api/visus/datasets/dyamond/T/visus.idx"),
    "theta": DatasetSpec("theta", "ocean", "sea-surface temperature", "mit_output/llc2160_theta/llc2160_theta.idx"),
    "u": DatasetSpec("u", "ocean", "east-west velocity", "mit_output/llc2160_arco/visus.idx"),
    "v": DatasetSpec("v", "ocean", "north-south velocity", "mit_output/llc2160_v/v_llc2160_x_y_depth.idx"),
    "salt": DatasetSpec("salt", "ocean", "salinity", "mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"),
}

def available_variables(domain=None):
    specs = list(_DATASET_SPECS.values())
    if domain is not None:
        specs = [s for s in specs if s.domain == domain]
    return [s.variable for s in specs]


def available_domains():
    return sorted({spec.domain for spec in _DATASET_SPECS.values()})

def is_atmospheric(variable):
    return get_dataset_spec(variable).domain == "atmosphere"


def is_oceanic(variable):
    return get_dataset_spec(variable).domain == "ocean"


def get_dataset_spec(variable):
    key = variable.strip()
    if key not in _DATASET_SPECS:
        valid = ", ".join(available_variables())
        raise ValueError(f"Unknown variable {variable!r}. Valid variables: {valid}")
    return _DATASET_SPECS[key]


def open_dataset(variable):
    spec = get_dataset_spec(variable)

    try:
        import OpenVisus as ov
    except Exception as exc:
        raise ImportError(
            "OpenVisus could not be imported. Install it in the active environment first."
        ) from exc

    db = ov.LoadDataset(spec.url)
    if db is None:
        raise RuntimeError(f"OpenVisus returned None for dataset {spec.url}")

    return db


def dataset_summary(db):
    logic_box = db.getLogicBox()
    timesteps = db.getTimesteps()
    field = db.getField()

    p1, p2 = tuple(logic_box[0]), tuple(logic_box[1])

    return {
        "url": db.getUrl(),
        "logic_box": (p1, p2),
        "shape": tuple(int(b - a) for a, b in zip(p1, p2)),
        "dims": len(p1),
        "timesteps": list(timesteps),
        "n_timesteps": len(timesteps),
        "field_name": field.name,
    }


def read_data(db, *, time=None, quality=None, x=None, y=None, z=None):
    kwargs = {}
    if time is not None:
        kwargs["time"] = int(time)
    if quality is not None:
        kwargs["quality"] = int(quality)
    if x is not None:
        kwargs["x"] = list(x)
    if y is not None:
        kwargs["y"] = list(y)
    if z is not None:
        if isinstance(z, int):
            kwargs["z"] = [z, z + 1]
        else:
            kwargs["z"] = list(z)

    return db.read(**kwargs)
    