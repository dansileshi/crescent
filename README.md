<div align="center">

# CRESCENT

**CRESCENT estimates condition-specific risks of first recorded long-term conditions**

A neural competing-risk model that turns a primary-care diagnosis history into<br>
condition-specific cumulative-incidence estimates, accounting for competing death.

[![Paper](https://img.shields.io/badge/paper-under%20review-lightgrey)](#citation)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24497210.svg)](https://doi.org/10.5281/zenodo.24497210)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](requirements.txt)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](requirements.txt)

[Manuscript release v1.0.1](https://github.com/dansileshi/crescent/releases/tag/v1.0.1) · [Model details](docs/model.md) · [Comparators](docs/comparators.md) · [Figures](figures/README.md) · [InflAIM codelists](https://github.com/Inflaim-Study/Inflaim-LTC-code-list) · Part of the [InflAIM programme](https://github.com/Inflaim-Study)

</div>

> **Status:** the paper is under review. This repository shares the full analysis code: the CRESCENT model, all comparator models, the evaluation and explanation analyses, and the figure-plotting code. No UK Biobank data or fitted model weights are shared.

<p align="center">
  <img src="figures/figure1/crescent_overview_animated.svg" alt="CRESCENT overview: a health record is encoded, read separately for each condition, combined into additive log-hazards, and converted into a competing-risk profile" width="100%">
</p>

---

## Overview

CRESCENT uses primary-care diagnosis histories and recruitment covariates to estimate the probability of a **first recorded primary-care diagnosis before death** for 58 long-term conditions from the [InflAIM framework](https://github.com/Inflaim-Study/Inflaim-LTC-code-list). Of the 60 prespecified conditions, two had fewer than 100 training cases and were not fitted.

We developed and evaluated the model in 229,950 UK Biobank participants with linked primary-care records, split by assessment centre into training (161,020), validation (22,962) and held-out test (45,968) sets. Predictors are fixed at recruitment, and follow-up starts six months later for participants who are alive, still under observation and free of the condition. Death is treated as a competing event.

We primarily evaluated **two- and five-year risks** (months 6–30 and 6–66 after recruitment), with one- and ten-year risks as secondary horizons. Follow-up is limited at ten years, so those estimates should be read with caution.

## Key findings

- CRESCENT reached a macro AUROC of **0.78** at two and five years and a five-year competing-risk concordance of **0.77**, tied with the locally pretrained CRESCENT-PT among nine models.
- At five years, CRESCENT and penalised Cox were within 0.01 AUROC for the ten most common conditions; several rarer conditions favoured CRESCENT.
- Diagnosis history accounted for the largest share of the model's predictions (**56%** of Shapley attribution), followed by age (19%) and BMI (10%).
- Removing event history from the disease rates lowered five-year macro AUROC by **0.131** in the ablation study.

## Repository

```
crescent/            CRESCENT model and training (paper §2.4–2.5)
evaluation/          Test metrics and bootstrap confidence intervals (§2.7, Figs. 2–3)
comparators/         The eight comparator models (§2.6, Fig. 2)
explainability/      Shapley, accumulated local effects and diagnosis-removal analyses (§2.9, Figs. 4–5)
supplementary/       Hyperparameter screen, history sensitivity and scaling (§2.8, Figs. S2–S4)
data_preparation/    Builds the prepared splits from UK Biobank records (§2.1–2.3)
figures/             Figure 1 graphic and plotting code for Figures 2–5
docs/                Model details, input format and comparator descriptions
run_crescent.py      Runs the CRESCENT pipeline
```

The comparators are penalised Cox, XGBoost, SurvTRACE-aligned, Dynamic-DeepHit-inspired, MOTOR, Delphi-2M-inspired, frozen Gemma 4 E4B and CRESCENT-PT. The plotting code runs on your own evaluation outputs.

## Getting started

```bash
git clone --branch v1.0.1 https://github.com/dansileshi/crescent.git
cd crescent
python -m pip install -r requirements.txt

export CRESCENT_PREPARED_DATA_DIR=/path/to/prepared_splits
export EHR_CRESCENT_FINAL_OUTPUT=runs/CRESCENT_Final     # a new, empty folder
python run_crescent.py                                  # full pipeline
python run_crescent.py --stages 1-4                     # stop after training
```

Running the model needs approved UK Biobank access and the prepared `train.csv`, `val.csv` and `test.csv` files built with `data_preparation/`. The paper used Python 3.11.4, PyTorch 2.6.0 and CUDA 12.4 on a single GPU. The pipeline trains the model, selects the checkpoint by validation loss, evaluates the held-out test set and computes 200-resample participant bootstrap intervals. Run the stages through `run_crescent.py` rather than importing them, because they share one Python session like notebook cells. Input fields and settings are described in [docs/model.md](docs/model.md).

Outputs are derived from UK Biobank data, so keep them in approved storage.

## Data and model availability

The code is shared; the data and fitted weights are not. In line with UK Biobank access conditions (application **303849**), the repository contains no participant-level data, derived datasets, aggregate result tables or model weights. Reproducing the results requires approved [UK Biobank access](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) and retraining.

Outcomes are first *recorded* primary-care diagnoses, not biological onset. Testing used held-out assessment centres within UK Biobank, not an independent cohort, and 94.9% of participants were recorded as White. CRESCENT is a research model and needs external evaluation before any clinical use.

## Citation

Please cite the paper and the [software release](https://doi.org/10.5281/zenodo.24497210):

```bibtex
@article{asfaw2026crescent,
  title   = {CRESCENT estimates condition-specific risks of first recorded long-term conditions},
  author  = {Asfaw, Daniel S. and Zebin, Tahmina and Khondoker, Mizanur and
             Olaniran, Oyebayo Ridwan and Paria, Soumya Subhra and
             Lewin, Alex and Welch, Ailsa and
             Kulinskaya, Elena and Fox, Chris and MacGregor, Alexander J. and
             Aung, Min Hane},
  journal = {TBC},
  year    = {2026},
  doi     = {TBC}
}
```

Software citation metadata is in [`CITATION.cff`](CITATION.cff).

## Authors and funding

Daniel S Asfaw¹, Tahmina Zebin², Mizanur Khondoker¹, Oyebayo Ridwan Olaniran³, Soumya Subhra Paria³, Alex Lewin³, Ailsa Welch⁴, Elena Kulinskaya³, Chris Fox⁵, Alexander J MacGregor¹, Min Hane Aung¹

¹ University of East Anglia · ² Brunel University of London · ³ London School of Hygiene and Tropical Medicine · ⁴ Norwich Medical School, University of East Anglia · ⁵ NIHR HealthTech Research Centre, University of Exeter

Contact: Daniel Asfaw (d.asfaw@uea.ac.uk)

This research used the UK Biobank Resource under application 303849; we thank the participants and the study team. The work was funded by the National Institute for Health and Care Research (NIHR) through grant NIHR205461 for the InflAIM programme. The views expressed are those of the authors and not necessarily those of the NIHR or the Department of Health and Social Care.

## License

Code is released under the [MIT License](LICENSE).
