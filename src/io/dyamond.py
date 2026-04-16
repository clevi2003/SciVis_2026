from __future__ import annotations

from dataclasses import dataclass


DYAMOND_BASE_URL = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"


@dataclass(frozen=True)
class DatasetSpec:
    variable: str
    idx_path: str

    @property
    def url(self):
        return DYAMOND_BASE_URL + self.idx_path


_DATASET_SPECS = {
    "salt": DatasetSpec("salt", "mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"),
    "v": DatasetSpec("v", "mit_output/llc2160_v/v_llc2160_x_y_depth.idx"),
    "theta": DatasetSpec("theta", "mit_output/llc2160_theta/llc2160_theta.idx"),
    "w": DatasetSpec("w", "mit_output/llc2160_w/llc2160_w.idx"),
    "u": DatasetSpec("u", "mit_output/llc2160_arco/visus.idx"),
}


def available_variables():
    return list(_DATASET_SPECS.keys())


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
    