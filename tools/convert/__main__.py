"""Convert explicitly selected local weights into a NInfer v3 artifact."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import sys
from collections.abc import Mapping

from .official_recipes import RECIPES
from .pipeline import convert, discard
from .proposal import DEFAULT_RANKING, add_official_proposal
from .qwen3_5 import build_model
from . import qwen4_exp
from .recipe import Recipe
from .sources.gguf import GgufModel
from .sources.gguf_ternary import GGUFFile
from .sources.safetensors import SafetensorsSource
from .ternary import RECIPES as TERNARY_RECIPES


class SourceInputs(Mapping):
    """Named optional sources are opened only when a recipe or component requests one."""

    def __init__(self, base, paths, stack):
        self._sources = {"base": base}
        self._paths = dict(paths)
        self._stack = stack

    def __getitem__(self, name):
        if name not in self._sources:
            if name not in self._paths:
                raise ValueError(
                    f"selected recipe requires source {name!r}; provide --source {name}=PATH"
                )
            opener = (
                GGUFFile
                if Path(self._paths[name]).suffix == ".gguf"
                else SafetensorsSource
            )
            self._sources[name] = self._stack.enter_context(opener(self._paths[name]))
        return self._sources[name]

    def __iter__(self):
        return iter(dict.fromkeys((*self._sources, *self._paths)))

    def __len__(self):
        return len(set(self._sources) | set(self._paths))

    def provenance(self):
        return {
            name: {"path": str(source.path)} for name, source in self._sources.items()
        }


def _pairs(values, label):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{label} requires unique NAME=PATH entries")
        result[name] = Path(path)
    return result


def _function(value: str):
    recipes = {**RECIPES, **TERNARY_RECIPES}
    if value in recipes:
        return recipes[value]
    filename, separator, function = value.rpartition(":")
    # A Windows drive or path colon is not a function separator: one is only an identifier.
    if not separator or not function.isidentifier():
        filename, function = value, "configure"
    path = Path(filename).resolve()
    spec = importlib.util.spec_from_file_location("ninfer_user_recipe", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load recipe file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    result = getattr(module, function)
    if not callable(result):
        raise TypeError(f"{value}: recipe entry must be callable")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="primary checkpoint/config and default resources, or the first file of a GGUF",
    )
    parser.add_argument(
        "--subset",
        choices=sorted(qwen4_exp.SUBSETS),
        help="convert only a development subset of a GGUF model (not loadable as a model)",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="official name or Python file[:function]; defaults to configure in a file",
    )
    parser.add_argument(
        "--override",
        help="Python configuration applied after the selected recipe and proposal",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named source such as quantized, dflash, dflash2, or mtp (replaces the base MTP head)",
    )
    parser.add_argument(
        "--components",
        default="text",
        help="comma-separated text,vision,mtp,dflash,dflash2",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="override a final frontend resource",
    )
    parser.add_argument(
        "--proposal",
        action="store_true",
        help="include the existing indexed proposal head",
    )
    parser.add_argument("--proposal-rows", type=int, default=131072)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--name", help="public instance name saved in metadata")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--max-file-bytes", type=int, default=32_000_000_000)
    parser.add_argument(
        "--ngram-out",
        type=Path,
        help="GGUF: where to write the n-gram volume (default OUT + '.ngram'); any drive",
    )
    parser.add_argument(
        "--ngram-reuse",
        type=Path,
        help="GGUF: bind the artifact to this existing n-gram volume of the same table",
    )
    parser.add_argument(
        "--mtp-gguf",
        type=Path,
        help="GGUF: add the MTP drafter from this GGUF (Strata's mtp-q2_0.gguf), component mtp",
    )
    parser.add_argument(
        "--mmproj",
        type=Path,
        help="GGUF: add the vision tower from this mmproj GGUF (image input), component vision",
    )
    args = parser.parse_args(argv)
    components = tuple(args.components.split(","))
    if len(components) != len(set(components)):
        raise ValueError("components must not repeat")
    paths = _pairs(args.source, "source")
    if "base" in paths:
        raise ValueError("select the base source with --model")
    overrides = _pairs(args.resource, "resource")
    gguf = args.model.suffix.lower() == ".gguf"
    if args.subset and not gguf:
        raise ValueError("--subset applies to a GGUF model")
    if gguf and (components != ("text",) or paths or args.proposal):
        raise ValueError("a GGUF model supplies only the text component and takes no --source")
    if not gguf and (args.ngram_out or args.ngram_reuse or args.mtp_gguf or args.mmproj):
        raise ValueError("--ngram-out, --ngram-reuse, --mtp-gguf and --mmproj apply to a GGUF model")
    if args.ngram_out and args.ngram_reuse:
        raise ValueError("--ngram-reuse and --ngram-out are exclusive")
    ngram_out = None
    with ExitStack() as stack:
        if gguf:
            base = stack.enter_context(GgufModel(args.model))
            sources = {"base": base}
            mtp = stack.enter_context(GgufModel(args.mtp_gguf)) if args.mtp_gguf else None
            mmproj = stack.enter_context(GgufModel(args.mmproj)) if args.mmproj else None
            model = qwen4_exp.build_model(
                base,
                subset=qwen4_exp.SUBSETS[args.subset] if args.subset else None,
                resource_overrides=overrides,
                mtp=mtp,
                vision=mmproj,
            )
            # The volume id ties the separately placed n-gram volume to this artifact.
            if args.ngram_reuse is not None:
                volume_id = qwen4_exp.read_ngram_volume_id(base, model, args.ngram_reuse)
            else:
                ngram_out = args.ngram_out or Path(str(args.out) + ".ngram")
                if ngram_out.exists():
                    raise FileExistsError(f"n-gram volume already exists: {ngram_out}")
                volume_id = os.urandom(16)
            qwen4_exp.bind_ngram_volume(model, volume_id)
        else:
            base = stack.enter_context(SafetensorsSource(args.model))
            sources = SourceInputs(base, paths, stack)
            companions = {
                key: sources[key] for key in ("dflash", "dflash2") if key in components
            }
            if "mtp" in components and "mtp" in paths:
                companions["mtp"] = sources["mtp"]
            model = build_model(
                base,
                components=components,
                companions=companions,
                resource_overrides=overrides,
            )
        recipe = Recipe(model)
        _function(args.recipe)(model, recipe, sources)
        if args.proposal:
            add_official_proposal(recipe, ranking=args.ranking, rows=args.proposal_rows)
        if args.override:
            _function(args.override)(model, recipe, sources)

        def progress(index, total, job):
            label = job.parameters[0]
            if len(job.parameters) > 1:
                label += f" (+{len(job.parameters)-1})"
            print(
                f"[{index+1}/{total}] {label}: {job.spec.format} {job.spec.shape}",
                flush=True,
            )

        provenance = {
            "converter": "ninfer-v3",
            "recipe": args.recipe,
            "sources": (
                {
                    "base": {"paths": [str(file.path) for file in base.files]},
                    **({"mtp": {"paths": [str(file.path) for file in mtp.files]}} if mtp is not None else {}),
                    **({"vision": {"paths": [str(file.path) for file in mmproj.files]}} if mmproj is not None else {}),
                }
                if gguf
                else sources.provenance()
            ),
        }
        if args.subset:
            provenance["subset"] = args.subset
        if args.override:
            provenance["override"] = args.override
        if args.proposal:
            provenance["ranking"] = str(args.ranking)
        report = convert(
            model,
            recipe,
            args.out,
            name=args.name,
            provenance=provenance,
            device=args.device,
            rows_per_chunk=args.rows_per_chunk,
            max_file_bytes=args.max_file_bytes,
            progress=progress,
        )
        print(
            f"wrote {args.out}: {report['objects']} objects, {len(report['files'])} files, {report['seconds']:.1f}s",
            flush=True,
        )
        if ngram_out is not None:
            # The artifact names a volume id that only this volume carries: without the volume
            # it is unusable, and it would block a retry at the same --out.
            try:
                geometry = qwen4_exp.write_ngram_volume(base, model, ngram_out)
            except BaseException:
                discard(report)
                raise
            print(f"wrote {ngram_out}: {geometry.blocks} blocks of {geometry.rows} rows", flush=True)
        elif gguf:
            print(f"bound {args.out} to the n-gram volume {args.ngram_reuse}", flush=True)


if __name__ == "__main__":
    main()
