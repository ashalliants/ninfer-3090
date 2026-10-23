#!/usr/bin/env python3
"""Keep the model cards in model-cards/ and the Hugging Face repos in step.

  python tools/sync_model_cards.py --check     verify each card's assets/ match docs/assets/
  python tools/sync_model_cards.py --publish   upload each card's README.md and assets/ to Hugging Face

A card's chart files are copies of docs/assets/<name>, so the Hugging Face page can load them from
its own repo. --check fails when a copy has drifted or has no source. --publish needs a token with
write access to the WarlaxZ organisation in HF_TOKEN.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "model-cards"
SOURCE_ASSETS = ROOT / "docs" / "assets"
HF_ORG = "WarlaxZ"
OWN_CARD_SUFFIX = "-NInfer-3090"  # the other cards describe upstream's repos, not ours
ASSET_REF = re.compile(r"""(?<![\w/.-])assets/([^\s)"'<>#?]+)""")  # assets/<file> in a README


def own_cards():
    return sorted(p for p in CARDS.iterdir() if p.is_dir() and p.name.endswith(OWN_CARD_SUFFIX))


def check():
    problems = []
    for card in own_cards():
        readme = (card / "README.md").read_text(encoding="utf-8")
        for name in sorted(set(ASSET_REF.findall(readme))):
            if not (card / "assets" / name).is_file():
                problems.append(f"{card.relative_to(ROOT)}/README.md: references assets/{name}, which is missing")
        for copy in sorted((card / "assets").glob("*")):
            source = SOURCE_ASSETS / copy.name
            if not source.is_file():
                problems.append(f"{copy.relative_to(ROOT)}: no source docs/assets/{copy.name}")
            elif source.read_bytes() != copy.read_bytes():
                problems.append(f"{copy.relative_to(ROOT)}: differs from docs/assets/{copy.name}")
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


def publish():
    for card in own_cards():
        repo = f"{HF_ORG}/{card.name}"
        print(f"uploading {card.relative_to(ROOT)} to {repo}", flush=True)
        subprocess.run(
            ["hf", "upload", repo, str(card), ".", "--repo-type", "model",
             "--include", "README.md", "--include", "assets/*",
             "--commit-message", "Sync model card from ashalliants/ninfer-3090"],
            check=True,
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    return check() if args.check else publish()


if __name__ == "__main__":
    sys.exit(main())
