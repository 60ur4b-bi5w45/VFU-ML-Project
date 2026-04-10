"""Extract matplotlib PNG outputs from notebooks into presentation_figures/ with slide filenames.

Zips, in order, each cell's plt.savefig('*.png') names with that cell's display_data image/png outputs.
"""
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "presentation_figures"
OUT.mkdir(exist_ok=True)

RENAME = {
    "gradient_mia_comparison": "01_gradient_mia_comparison",
    "gradient_feature_importance": "02_gradient_feature_importance",
    "fuma_results": "03_fuma_results",
    "mia_results": "04_mia_results",
    "shadow_attack_results": "05_shadow_attack_results",
    "comprehensive_verification": "06_comprehensive_verification",
    "phase1_validation_dashboard": "07_phase1_validation_dashboard",
    "training_history_adult": "08_training_history_adult",
}

SAVEFIG_RE = re.compile(r"plt\.savefig\s*\(\s*['\"]([a-zA-Z0-9_]+)\.png")


def extract_from_notebook(nb_path: Path, found: dict) -> int:
    written = 0
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        bases = [b for b in SAVEFIG_RE.findall(src) if b in RENAME]
        if not bases:
            continue
        pngs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") != "display_data":
                continue
            b64 = out.get("data", {}).get("image/png")
            if b64:
                pngs.append(base64.b64decode(b64))
        for base, raw in zip(bases, pngs):
            name = RENAME[base] + ".png"
            (OUT / name).write_bytes(raw)
            found[base] = True
            written += 1
    return written


def main():
    found = dict.fromkeys(RENAME, False)
    total = 0
    for nb_name in ("Part_1.ipynb", "VFL_Unlearning_Project.ipynb"):
        p = ROOT / nb_name
        if p.is_file():
            n = extract_from_notebook(p, found)
            total += n
            print(f"{nb_name}: extracted {n} image(s)")
    print(f"Total written to {OUT}/: {total}")
    for k in sorted(RENAME):
        ok = found.get(k, False)
        print(f"  {RENAME[k]}.png: {'ok' if ok else 'missing - run verification cells in notebook'}")


if __name__ == "__main__":
    main()
