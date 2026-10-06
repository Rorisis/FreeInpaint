import hashlib
import json
from pathlib import Path


def resolve_context_views(context, scene, object_names):
    requested = context.get(scene)
    if (
        not isinstance(requested, list)
        or len(requested) != 4
        or len(set(requested)) != 4
    ):
        raise ValueError(f"{scene}: context JSON must contain four distinct views")
    selected = []
    for stem in requested:
        matches = [name for name in object_names if Path(name).stem == Path(stem).stem]
        if len(matches) != 1:
            raise ValueError(
                f"{scene}: input {stem!r} must match exactly one OBJECT image; got {matches}"
            )
        selected.append(matches[0])
    return selected


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
