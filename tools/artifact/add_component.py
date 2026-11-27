"""Copy one speculative component (its config, objects, bindings and uses) from a donor artifact.

Writes a new artifact; neither input is modified. The base keeps every object, binding and use as
it is, so its own components decode exactly as before. The donor component's bindings must name
objects no other donor component uses, and its uses may reference the base only through
parameters both artifacts bind with the same tensor shape (for example ``text/output_head``).

    python3 -m tools.artifact.add_component BASE DONOR OUTPUT --component dflash2

The component's drafter was trained against the donor's target; acceptance with a different
target of the same architecture (another quantization of the same model) has to be measured.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .reader import Artifact
from .schema import ArtifactError, ResourceObject, ResourceSpec, TensorObject, TensorSpec
from .writer import ArtifactWriter


def _objects_of(binding: dict) -> set[str]:
    if "object" in binding:
        return {binding["object"]}
    return {part["object"] for part in binding.get("parts", ())}


def _rename(binding: dict, renamed: dict[str, str]) -> dict:
    if "object" in binding:
        return {**binding, "object": renamed[binding["object"]]}
    return {
        **binding,
        "parts": [{**part, "object": renamed[part["object"]]} for part in binding["parts"]],
    }


def _spec(obj, new_id: str):
    if isinstance(obj, TensorObject):
        return TensorSpec(new_id, obj.shape, obj.format, obj.layout)
    if isinstance(obj, ResourceObject):
        return ResourceSpec(new_id, obj.bytes, obj.encoding)
    raise TypeError(type(obj).__name__)


def _shape(artifact: Artifact, binding: dict) -> tuple[int, ...] | None:
    if "object" not in binding:
        return None
    obj = artifact.object(binding["object"])
    return obj.shape if isinstance(obj, TensorObject) else None


def add_component(base_path: Path, donor_path: Path, output: Path, component: str) -> None:
    prefix = component + "/"
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        b, d = base.directory, donor.directory
        if component in b.components:
            raise ArtifactError(f"{base_path} already has component {component!r}")
        if component not in d.components:
            raise ArtifactError(f"{donor_path} has no component {component!r}")
        target = d.components[component].get("target")
        if target is not None and target not in b.components:
            raise ArtifactError(f"the base has no {target!r} component for {component!r}")

        bindings = {k: v for k, v in d.bindings.items() if k.startswith(prefix)}
        owned: set[str] = set()
        for value in bindings.values():
            owned |= _objects_of(value)
        others: set[str] = set()
        for name, value in d.bindings.items():
            if not name.startswith(prefix):
                others |= _objects_of(value)
        if owned & others:
            raise ArtifactError(f"{component} shares objects with other donor components")

        uses = [
            use for use in d.uses
            if any(isinstance(v, str) and v.startswith(prefix) for v in use.values())
        ]
        for use in uses:
            parameter = use["parameter"]
            if parameter.startswith(prefix):
                continue
            if parameter not in b.bindings:
                raise ArtifactError(f"{component} uses {parameter}, which the base does not bind")
            if _shape(base, b.bindings[parameter]) != _shape(donor, d.bindings[parameter]):
                raise ArtifactError(f"{parameter}: shapes differ between base and donor")
            if "auxiliaries" in use:
                raise ArtifactError(f"{parameter}: a use with auxiliaries cannot be carried over")

        taken = {obj.id for obj in b.objects}
        renamed: dict[str, str] = {}
        donor_order = [obj for obj in d.objects if obj.id in owned]
        for obj in donor_order:
            new_id = f"{component}/{obj.id}"
            if new_id in taken:
                raise ArtifactError(f"object id {new_id} already exists in the base")
            renamed[obj.id] = new_id

        specs = [_spec(obj, obj.id) for obj in b.objects]
        specs += [_spec(obj, renamed[obj.id]) for obj in donor_order]
        provenance = dict(b.provenance)
        provenance["added_components"] = list(provenance.get("added_components", [])) + [
            {"component": component, "donor": donor_path.name,
             "donor_artifact_id": donor.artifact_id.hex()}
        ]
        writer = ArtifactWriter(
            output,
            specs,
            components={**b.components, component: d.components[component]},
            bindings={**b.bindings, **{k: _rename(v, renamed) for k, v in bindings.items()}},
            uses=[*b.uses, *uses],
            metadata=b.metadata,
            provenance=provenance,
        )
        with writer:
            for obj in b.objects:
                writer.write_object(obj.id, base.iter_object(obj.id))
            for obj in donor_order:
                writer.write_object(renamed[obj.id], donor.iter_object(obj.id))
            writer.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("donor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--component", required=True)
    args = parser.parse_args()
    add_component(args.base, args.donor, args.output, args.component)


if __name__ == "__main__":
    main()
