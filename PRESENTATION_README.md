# Presentation Figures and Slides

## Saving figures for the presentation

1. **Run the notebook** (`VFL_Unlearning_Project.ipynb`) from top to bottom (or at least run Phase 0, Phase 1 verification, and all visualization cells).
2. Every plot is saved **separately** into the folder **`presentation_figures/`** with numbered filenames:

   | File | Content |
   |------|---------|
   | `01_gradient_mia_comparison.png` | Output vs Gradient MIA + gradient change |
   | `02_gradient_feature_importance.png` | Gradient MIA feature importance |
   | `03_fuma_results.png` | FUMA verification (canary, detection rate) |
   | `04_mia_results.png` | Baseline MIA results |
   | `05_shadow_attack_results.png` | Shadow model attack |
   | `06_comprehensive_verification.png` | All metrics + pass/fail |
   | `07_phase1_validation_dashboard.png` | Phase 1 dashboard (4 panels) |
   | `08_training_history_adult.png` | Training curves |

3. Each figure is also saved once in the project root (e.g. `gradient_mia_comparison.png`) for quick reference.

## LaTeX presentation

- **File:** `evfu_presentation.tex` (Beamer).
- **Location:** Keep `evfu_presentation.tex` in the **same directory** as the `presentation_figures` folder (project root).
- **Compile:**
  ```bash
  pdflatex evfu_presentation.tex
  ```
  (Run twice if needed for TOC.)
- The slides use `\graphicspath{{./presentation_figures/}}`, so they include the numbered PNGs from that folder. One slide per main figure for easy explanation.

## If a figure is missing

- Ensure the corresponding notebook section has been run (e.g. run Phase 1 verification before `07_phase1_validation_dashboard.png`).
- The notebook creates `presentation_figures/` automatically on first run (in the setup cell).
