# Perception-to-Imagery Transfer: EEG Decoding

Research code for investigating whether EEG decoders trained on perceptual activity can generalize to mental imagery, including comparisons across sensory modalities and participants.

## Overview

This project examines the transfer of EEG-based neural decoding between perception and mental imagery. The main questions are:

- Can patterns learned from perception be used to decode corresponding mental imagery?
- How does decoder performance change across sensory modalities and participants?
- How do different decoding models and evaluation settings affect transfer performance?

The repository contains experimental evaluation scripts and supporting code used in this research. It is shared primarily as an example of research and programming work, rather than as a ready-to-run software package.

## Code organization

| Path | Purpose |
| --- | --- |
| `modules/` | Supporting model definitions and evaluation utilities |
| `notebooks/` | Research notebooks and exploratory analyses |
| `evsub.py` | Main subject-level decoding evaluation entry point |
| `evsubmix.py` | Alternative evaluation script, including confusion-matrix outputs |
| `evsub.sh` | SLURM job-array script for `evsub.py` |
| `evsubmix.sh` | SLURM script for the alternative evaluation |

The evaluation scripts expose model choices including logistic regression, xDAWN-based decoding, EEGNet, EEGConformer, AttentionBaseNet, and EEGNeX (depending on the supporting model implementation).

The scripts include within-subject and cross-subject evaluation settings, and process two-class and four-class tasks. The `evsubmix.py` workflow saves normalized confusion matrices for selected evaluations.

## Author

Zichen Song — Graduate School of Informatics, Kyoto University  
