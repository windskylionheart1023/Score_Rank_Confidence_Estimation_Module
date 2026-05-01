#!/usr/bin/env python3
"""Build a word-level SR-CEM training/test dataset from ESPnet decode outputs.

For each word w_l of the 1-best hypothesis we extract the SR-CEM word
features described in Eq. 14 of the paper:

    f_W(w_l) = [ s_W(w_l), r_l(w_l), S_<l, S_>l, Q_l ]

i.e. (word score = sum of token scores, max token-rank within the word,
preceding-cumulative score, succeeding-cumulative score, token count Q_l).
The mean softmax token confidence is also kept as feature index 0 for
diagnostic comparison.

Inputs (per decode directory): same as :mod:`dataset_token_srcem`.
"""

import argparse
import json
import os
import re

import numpy as np
import torch

from espnet2.asr.CEM.model import NamedTensorDataset
from espnet2.text.token_id_converter import TokenIDConverter


FEATURE_NAMES = [
    "confidence",
    "score",
    "max_rank",
    "prev_score_sum",
    "after_score_sum",
    "token_number_per_word",
]


def parse_file_to_dict(filepath: str) -> dict:
    """Parse ``token_int``-style files into ``{utt_id: tokens}``.

    Tries several utt-id regex flavors (LibriSpeech, CommonVoice, Libri-Adapt,
    CGN) so the same parser can be reused across the datasets in the paper.
    """
    id_patterns = [
        re.compile(r"^\d+-\d+-\d+"),
        re.compile(r"^[a-zA-Z0-9]+-common_voice_en_\w+"),
        re.compile(r"^[a-z]{2}_[a-z]+/[a-z]+_[a-z]+_test_\d+_\d+-\d+"),
        re.compile(r"fv\d+-\d+"),
        re.compile(r"^[a-z]{2}_[a-z]{2}(?:_[a-z]+)*_\d+_\d+-\d+"),
    ]

    result = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = next((p.match(line) for p in id_patterns if p.match(line)), None)
            if not match:
                continue

            utt_id = match.group(0)
            content = line[len(utt_id):].strip().split()
            result[utt_id] = [int(x) if x.isdigit() else x for x in content]
    return result


def process_file_to_dict(input_path: str) -> dict:
    """Parse an ESPnet ``result.txt`` (sclite-style) into ``{utt_id: blocks}``."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Speaker sentences" in line:
            lines = lines[i:]
            break

    data = {}
    for i, line in enumerate(lines):
        if line.strip().startswith("id: ("):
            parts = line.split()
            if len(parts) < 2 or i + 4 >= len(lines):
                continue
            utt_id = parts[1].strip("()")
            data[utt_id] = {
                "Scores": lines[i + 1].rstrip(),
                "REF": lines[i + 2].replace("REF:", "").strip().split(),
                "HYP": lines[i + 3].replace("HYP:", "").strip().split(),
                "Eval": lines[i + 4].rstrip(),
            }
    return data


def _label_id_for(score_id: str) -> str:
    if "common_voice" in score_id or "en_us_" in score_id:
        return f"{score_id.split('-')[0]}-{score_id}"
    if "fv" in score_id:
        return f"{score_id}-spk-{score_id}"
    parts = score_id.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}-{score_id}"
    if len(parts) == 2:
        return f"{parts[0]}-{score_id}"
    return score_id


def process_data(label_file: str, logdir: str, token_path: str, n: int, transducer: bool):
    """Aggregate per-token scores into per-word feature lists per utterance."""
    if not os.path.isfile(label_file):
        print(f"Label file not found: {label_file}")
        return {}

    label_dict = process_file_to_dict(label_file)

    scores_path = os.path.join(logdir, f"output.1/{n}best_recog/scores_list.json")
    if not os.path.isfile(scores_path):
        print(f"Scores file not found: {scores_path}")
        return {}
    with open(scores_path) as f:
        scores_dict = json.load(f)

    hyp_path = os.path.join(logdir, f"output.1/{n}best_recog/token_int")
    if not os.path.isfile(hyp_path):
        print(f"Token_int file not found: {hyp_path}")
        return {}
    hyp_int_dict = parse_file_to_dict(hyp_path)

    token_id_converter = TokenIDConverter(token_path)

    result = {}
    skipped = 0

    for score_id, all_score in scores_dict:
        label_id = _label_id_for(score_id)
        if label_id not in label_dict:
            skipped += 1
            continue

        token_label = [t for t in label_dict[label_id]["REF"] if "*" not in t]
        token_hyp = [t for t in label_dict[label_id]["HYP"] if "*" not in t]
        hyp_int = hyp_int_dict.get(score_id, [])
        hyp_tokens = token_id_converter.ids2tokens(hyp_int)

        if (
            not hyp_int
            or len(token_hyp) != len(hyp_int)
            or len(token_label) != len(token_hyp)
            or len(all_score) != len(hyp_int)
        ):
            skipped += 1
            continue

        # Word boundaries marked by the leading ▁ subword character.
        word_boundary = [i for i, tok in enumerate(hyp_tokens) if tok.startswith("▁")]
        word_boundary.append(len(hyp_tokens))
        if word_boundary and word_boundary[0] != 0:
            word_boundary.insert(0, 0)

        # Per-token stats (cumulative score, rank, softmax confidence).
        accu_scores, ranks, confidences = [], [], []
        for i, hid in enumerate(hyp_int):
            key = str(hid - 1 if transducer else hid)
            accu_scores.append(all_score[i][key])
            ranks.append(list(all_score[i].keys()).index(key))

            values = np.array(list(all_score[i].values()))
            probs = np.exp(values - values.max())
            probs /= probs.sum()
            confidences.append(dict(zip(all_score[i].keys(), probs))[key])

        # Differentiate cumulative beam scores into per-step increments.
        scores = [accu_scores[0]] + [
            c - p for p, c in zip(accu_scores, accu_scores[1:])
        ]

        word_scores = [scores[word_boundary[i]:word_boundary[i + 1]] for i in range(len(word_boundary) - 1)]
        word_conf = [confidences[word_boundary[i]:word_boundary[i + 1]] for i in range(len(word_boundary) - 1)]
        word_ranks = [ranks[word_boundary[i]:word_boundary[i + 1]] for i in range(len(word_boundary) - 1)]

        result[score_id] = {
            "scores": [sum(block) for block in word_scores],
            "confidence": [np.mean(block) if block else 0.0 for block in word_conf],
            "max_rank": [np.max(block) if block else 0.0 for block in word_ranks],
            "min_rank": [np.min(block) if block else 0.0 for block in word_ranks],
            "mean_rank": [np.mean(block) if block else 0.0 for block in word_ranks],
            "token_number_per_word": [len(block) for block in word_scores],
        }

    print(f"Word-level: processed {len(scores_dict)} utterances, skipped {skipped}.")
    return result


def get_word_data(decode_dir: str, word_info_dict: dict):
    """Combine per-word features with WER-derived correctness labels."""
    word_correctness_path = os.path.join(decode_dir, "score_wer/result.txt")
    if not os.path.isfile(word_correctness_path):
        print(f"Missing word correctness file: {word_correctness_path}")
        return torch.empty(0), torch.empty(0)

    word_correctness_dict = process_file_to_dict(word_correctness_path)

    full_conf, full_score, full_max_rank = [], [], []
    full_prev_sum, full_after_sum, full_token_num = [], [], []
    correctness = []

    for utt_id, feats in word_info_dict.items():
        label_id = _label_id_for(utt_id)
        if label_id not in word_correctness_dict:
            continue

        label = [t for t in word_correctness_dict[label_id]["REF"] if "*" not in t]
        hyp = [t for t in word_correctness_dict[label_id]["HYP"] if "*" not in t]

        if len(label) != len(feats["confidence"]):
            continue

        for ref_word, hyp_word, conf in zip(label, hyp, feats["confidence"]):
            correctness.append(1 if ref_word.upper() == hyp_word.upper() else 0)
            full_conf.append(conf)

        scores = feats["scores"]
        full_score.extend(scores)
        full_max_rank.extend(feats["max_rank"])
        full_token_num.extend(feats["token_number_per_word"])

        prev_sum = [0] + [sum(scores[:i]) for i in range(1, len(scores))]
        after_sum = [
            sum(scores[i + 1:]) if i < len(scores) - 1 else 0
            for i in range(len(scores))
        ]
        full_prev_sum.extend(prev_sum)
        full_after_sum.extend(after_sum)

    data = torch.tensor(
        list(zip(full_conf, full_score, full_max_rank, full_prev_sum, full_after_sum, full_token_num))
    ).float()
    target = torch.tensor(correctness).unsqueeze(1).float()
    return data, target


def create_dataset(
    decode_path: str,
    token_path: str,
    split: str,
    output_dir: str,
    suffix: str,
    transducer: bool,
):
    logdir = os.path.join(decode_path, "logdir")
    word_info = process_data(
        os.path.join(decode_path, "score_ter/result.txt"),
        logdir,
        token_path,
        n=1,
        transducer=transducer,
    )
    data, target = get_word_data(decode_path, word_info)

    assert data.shape[1] == len(FEATURE_NAMES), "Feature/name length mismatch"
    dataset = NamedTensorDataset(data, target, feature_names=FEATURE_NAMES)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{split}_dataset_word{suffix}.pt")
    torch.save(dataset, save_path)
    print(f"Saved {split} word dataset to {save_path} (N={len(data)})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-path", required=True, help="ESPnet decode directory.")
    parser.add_argument("--token-path", required=True, help="BPE tokens.txt file.")
    parser.add_argument("--output-dir", required=True, help="Where to save the .pt dataset.")
    parser.add_argument("--split", default="train", help="Split tag used in output filename.")
    parser.add_argument("--suffix", default="", help="Optional suffix appended to output filename.")
    parser.add_argument(
        "--transducer",
        action="store_true",
        help="Set if the ASR is RNN-T (token ids are 1-indexed in the score dict).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(
        decode_path=args.decode_path,
        token_path=args.token_path,
        split=args.split,
        output_dir=args.output_dir,
        suffix=args.suffix,
        transducer=args.transducer,
    )
