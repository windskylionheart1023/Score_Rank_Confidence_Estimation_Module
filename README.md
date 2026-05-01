# SR-CEM: Score-Rank Confidence Estimation Module

Reference implementation of:

> **Leveraging Beam Search Information for Confidence Estimation in E2E ASR.**
> Yichen Jia, Hugo Van hamme. *IEEE Open Journal of Signal Processing*.

SR-CEM is a lightweight, **architecture-agnostic** confidence-estimation
module for end-to-end ASR. It predicts token- and word-level confidence
scores from quantities that are already available during beam search:
the per-step token score, the token's rank in the score vector, the
preceding/succeeding cumulative scores, and the top-K score values.

Trained as a single hidden-layer MLP (~0.4-0.6k parameters), SR-CEM
substantially reduces *Maximum Calibration Error* — the worst-case
calibration gap that matters for downstream decisions — across hybrid
CTC/Attention, attention-only, CTC-only, and RNN-T architectures, on
LibriSpeech, Common Voice, Libri-Adapt, CGN (Dutch), and CHiME-6.

## Repository layout

```
.
|-- README.md
`-- espnet/                                      <- mirrors the ESPnet tree
    |-- espnet/nets/
    |   |-- beam_search.py                       (patched: scores_list field)
    |   `-- batch_beam_search.py                 (patched: carry scores_list)
    `-- espnet2/
        |-- asr/
        |   |-- AdaptiveBinning.py               adaptive-binning ECE/MCE
        |   |-- ctc.py                           (forked from ESPnet)
        |   |-- transducer/beam_search_transducer.py  (patched: scores_list)
        |   `-- CEM/
        |       |-- model.py                     Score_CEM + baselines
        |       |-- metrics.py                   NCE / AUC-ROC / AUC-PR
        |       |-- trueclass_model.py           TruCLeS baselines
        |       |-- dataset_token_srcem.py       build token-level dataset
        |       |-- dataset_word_srcem.py        build word-level dataset
        |       |-- train_token_srcem.py         train token SR-CEM
        |       |-- train_word_srcem.py          train word SR-CEM
        |       |-- test_token_srcem.py          evaluate token SR-CEM
        |       `-- test_word_srcem.py           evaluate word SR-CEM
        `-- bin/
            |-- asr_inference.py                 (patched: dump scores_list.json)
            `-- asr_align.py                     (forked from ESPnet)
```

## Installation

The pipeline assumes a working [ESPnet](https://github.com/espnet/espnet)
checkout. To use SR-CEM, copy the files under [espnet/](espnet/) into the
matching paths inside your ESPnet installation (the directory layout
mirrors ESPnet exactly), overwriting the upstream files.

```bash
# from inside this repo:
cp -r espnet/espnet  /path/to/espnet/espnet
cp -r espnet/espnet2 /path/to/espnet/espnet2
```

The patched modules are functionally backwards-compatible: they only
*append* a `scores_list` field to `Hypothesis` and a JSON dump after
inference. Standard ESPnet recipes continue to run unchanged.

Python dependencies (in addition to ESPnet's own): `torch`, `numpy`,
`scikit-learn`, `matplotlib`.

## Pipeline

### 1. Decode with the patched ESPnet

Run any ESPnet ASR recipe through inference normally. Because
`asr_inference.py` is patched, the decode directory will additionally
contain:

```
<decode_dir>/
|-- score_wer/result.txt
|-- score_ter/result.txt
`-- logdir/output.1/{n}best_recog/
    |-- token_int
    `-- scores_list.json          <-- new: per-step score vectors
```

### 2. Build SR-CEM datasets

Token-level features (8-dim, see Eq. 9 in the paper):

```bash
python -m espnet2.asr.CEM.dataset_token_srcem \
    --decode-path /path/to/<decode_dir> \
    --token-path  /path/to/tokens.txt \
    --output-dir  /path/to/dataset \
    --split       train
```

Word-level features (5-dim, see Eq. 14):

```bash
python -m espnet2.asr.CEM.dataset_word_srcem \
    --decode-path /path/to/<decode_dir> \
    --token-path  /path/to/tokens.txt \
    --output-dir  /path/to/dataset \
    --split       train
```

For the RNN-T setup add `--transducer`. Repeat with `--split test` (and
the corresponding test decode directory) to build the test dataset.

### 3. Train SR-CEM

```bash
python -m espnet2.asr.CEM.train_token_srcem \
    --train-dataset /path/to/dataset/train_dataset_token.pt \
    --model-out     /path/to/model/cem_token.pt \
    --loss-plot     /path/to/loss/cem_token.png \
    --num-epochs 20 --learning-rate 1e-3 --weight-decay 1e-4
```

Same shape for `train_word_srcem`. The default `--remove-features
confidence` keeps the raw softmax confidence out of the input (it is only
used as a diagnostic baseline at evaluation time).

### 4. Evaluate

```bash
python -m espnet2.asr.CEM.test_token_srcem \
    --test-dataset /path/to/dataset/test_dataset_token.pt \
    --model-path   /path/to/model/cem_token.pt \
    --cem-plot     /path/to/plot/rd_cem_token.png \
    --native-plot  /path/to/plot/rd_softmax_token.png
```

Reported metrics: NCE, AUC-ROC, AUC-PR, AUC-PR (errors as positives), and
adaptive-binning ECE/MCE. Reliability diagrams are saved to the paths
provided.

`test_word_srcem` is invoked the same way with the word-level dataset.

## Headline results (paper, Tabs. 2-3)

In-domain LibriSpeech test-clean, hybrid CTC/Attention backbone:

| Level | Method        | NCE   | AUC-ROC | AUC-PR | ECE (%) | MCE (%) |
|-------|---------------|-------|---------|--------|---------|---------|
| Token | Softmax       | 0.301 | 0.919   | 0.996  | 1.75    | 20.04   |
| Token | **SR-CEM**    | 0.383 | 0.923   | 0.996  | 0.30    | **4.50**  |
| Word  | Softmax       | 0.336 | 0.931   | 0.996  | 1.67    | 17.91   |
| Word  | **SR-CEM**    | 0.356 | 0.899   | 0.994  | 0.35    | **8.17**  |

SR-CEM uses ~0.6k (token) / ~0.4k (word) parameters - 100-250x smaller
than the MLP/Transformer baselines while reducing MCE by 50-70%.

## Citation

```bibtex
@article{jia2025srcem,
  title   = {Leveraging Beam Search Information for Confidence Estimation
             in E2E ASR},
  author  = {Jia, Yichen and Van hamme, Hugo},
  journal = {IEEE Open Journal of Signal Processing},
  year    = {2025}
}
```

## Acknowledgements

This work was supported by the Flemish Government under the FWO-SBO grant
S004923N: NELF, and KU Leuven grant C24M/22/025.

The patched ESPnet modules are derivative works of
[ESPnet](https://github.com/espnet/espnet) and inherit ESPnet's Apache 2.0
license.
