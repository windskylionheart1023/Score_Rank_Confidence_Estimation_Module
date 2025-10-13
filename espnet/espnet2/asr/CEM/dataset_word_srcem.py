#!/usr/bin/env python3
import os
import re
import json
import numpy as np
import torch
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.asr.CEM.model import NamedTensorDataset


###############################################################################
#                                CONFIG                                       #
###############################################################################
SUFFIX = "_ours_libapt_s4"
TRANSDUCER = False
NBEST = 1
CEM_PATH = "/esat/audioslave/yjia/espnet/espnet/espnet2/asr/CEM"

FEATURE_NAMES = [
    "confidence", "score", "max_rank",
    "prev_score_sum", "after_score_sum",
    "token_number_per_word"
]


###############################################################################
#                          SHARED HELPER FUNCTIONS                            #
###############################################################################
def parse_file_to_dict(filepath: str) -> dict:
    """
    Parses text-like files into dict[id] -> [tokens].
    Supports multiple ID formats (LibriSpeech, CommonVoice, CGN, etc.).
    """
    id_patterns = [
        re.compile(r"^\d+-\d+-\d+"),  # LibriSpeech style
        re.compile(r"^[a-zA-Z0-9]+-common_voice_en_\w+"),
        re.compile(r"^[a-z]{2}_[a-z]+/[a-z]+_[a-z]+_test_\d+_\d+-\d+"),
        re.compile(r"fv\d+-\d+"),  # CGN
        re.compile(r"^[a-z]{2}_[a-z]{2}(?:_[a-z]+)*_\d+_\d+-\d+"),
    ]

    result = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = None
            for pat in id_patterns:
                match = pat.match(line)
                if match:
                    break
            if not match:
                continue

            utt_id = match.group(0)
            content = line[len(utt_id):].strip().split()
            content = [int(x) if x.isdigit() else x for x in content]
            result[utt_id] = content
    return result


def process_file_to_dict(input_path: str) -> dict:
    """
    Parses result.txt into dict[id] -> {"REF": [...], "HYP": [...], ...}.
    """
    with open(input_path, "r") as f:
        lines = f.readlines()

    # keep only part after "Speaker sentences"
    for i, line in enumerate(lines):
        if "Speaker sentences" in line:
            lines = lines[i:]
            break

    data = {}
    for i, line in enumerate(lines):
        if line.strip().startswith("id: ("):
            parts = line.split()
            if len(parts) < 2:
                continue
            utt_id = parts[1].strip("()")

            if i + 4 < len(lines):
                values = {
                    "Scores": lines[i + 1].rstrip(),
                    "REF": lines[i + 2].replace("REF:", "").strip().split(),
                    "HYP": lines[i + 3].replace("HYP:", "").strip().split(),
                    "Eval": lines[i + 4].rstrip(),
                }
                data[utt_id] = values
    return data


###############################################################################
#                         TOKEN / WORD PROCESSING                             #
###############################################################################
def process_data(label_file, logdir, token_path, n) -> dict:
    """
    Collects token- and word-level calibration features.
    Returns dict[utt_id] -> per-word feature lists.
    """
    if not os.path.isfile(label_file):
        print(f"Label file not found: {label_file}")
        return {}

    label_dict = process_file_to_dict(label_file)

    scores_path = os.path.join(logdir, f"output.1/{n}best_recog/scores_list.json")
    if not os.path.isfile(scores_path):
        print(f"Scores file not found: {scores_path}")
        return {}
    with open(scores_path, "r") as f:
        scores_dict = json.load(f)

    hyp_path = os.path.join(logdir, f"output.1/{n}best_recog/token_int")
    if not os.path.isfile(hyp_path):
        print(f"Token_int file not found: {hyp_path}")
        return {}
    hyp_int_dict = parse_file_to_dict(hyp_path)

    tokenidconvertor = TokenIDConverter(token_path)

    result = {}
    skipped = 0

    for score_id, all_score in scores_dict:
        # construct label_id
        if "common_voice" in score_id or "en_us_" in score_id:
            label_id = f"{score_id.split('-')[0]}-{score_id}"
        elif "fv" in score_id:
            label_id = f"{score_id}-spk-{score_id}"
        else:
            parts = score_id.split("-")
            label_id = f"{parts[0]}-{parts[1]}-{score_id}"

        if label_id not in label_dict:
            skipped += 1
            continue

        token_label = [t for t in label_dict[label_id]["REF"] if "*" not in t]
        token_hyp = [t for t in label_dict[label_id]["HYP"] if "*" not in t]
        hyp_int = hyp_int_dict.get(score_id, [])
        hyp_tokens = tokenidconvertor.ids2tokens(hyp_int)

        if (
            not hyp_int
            or len(token_hyp) != len(hyp_int)
            or len(token_label) != len(token_hyp)
            or len(all_score) != len(hyp_int)
        ):
            skipped += 1
            continue

        # find word boundaries
        word_boundary = [i for i, tok in enumerate(hyp_tokens) if tok.startswith("▁")]
        word_boundary.append(len(hyp_tokens))
        if word_boundary and word_boundary[0] != 0:
            word_boundary.insert(0, 0)

        # per-token stats
        accu_scores, ranks, confidences = [], [], []
        for i, hid in enumerate(hyp_int):
            key = str(hid - 1 if TRANSDUCER else hid)
            accu_scores.append(all_score[i][key])
            ranks.append(list(all_score[i].keys()).index(key))
            values = np.array(list(all_score[i].values()))
            probs = np.exp(values - values.max())
            probs /= probs.sum()
            confidences.append(dict(zip(all_score[i].keys(), probs))[key])

        # convert to increments
        scores = [accu_scores[0]] + [c - p for p, c in zip(accu_scores, accu_scores[1:])]

        # split into words
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

    print(f"Processed {len(scores_dict)} entries, skipped {skipped}.")
    return result


def get_word_data(decode_dir: str, word_info_dict: dict):
    """
    Combines per-word features with correctness labels from WER results.
    Returns: (data_tensor, target_tensor).
    """
    word_correctness_path = os.path.join(decode_dir, "score_wer/result.txt")
    if not os.path.isfile(word_correctness_path):
        print("Missing word correctness file:", word_correctness_path)
        return torch.empty(0), torch.empty(0)

    word_correctness_dict = process_file_to_dict(word_correctness_path)

    full_conf, full_score, full_max_rank = [], [], []
    full_prev_sum, full_after_sum, full_token_num = [], [], []
    correctness = []

    for utt_id, feats in word_info_dict.items():
        scores = feats["scores"]
        confs = feats["confidence"]
        max_ranks = feats["max_rank"]
        token_num = feats["token_number_per_word"]

        # build label_id to match WER dict
        parts = utt_id.split("-")
        if "fv" in utt_id:
            label_id = f"{utt_id}-spk-{utt_id}"
        elif len(parts) >= 3:
            label_id = f"{parts[0]}-{parts[1]}-{utt_id}"
        elif len(parts) == 2:
            label_id = f"{parts[0]}-{utt_id}"
        else:
            label_id = utt_id

        if label_id not in word_correctness_dict:
            continue
        label = [t for t in word_correctness_dict[label_id]["REF"] if "*" not in t]
        hyp = [t for t in word_correctness_dict[label_id]["HYP"] if "*" not in t]

        if len(label) != len(confs):
            continue

        # correctness per word
        for l, h, c in zip(label, hyp, confs):
            correctness.append(1 if l.upper() == h.upper() else 0)
            full_conf.append(c)

        full_score.extend(scores)
        full_max_rank.extend(max_ranks)
        full_token_num.extend(token_num)

        prev_sum = [0] + [sum(scores[:i]) for i in range(1, len(scores))]
        after_sum = [sum(scores[i + 1:]) if i < len(scores) - 1 else 0 for i in range(len(scores))]
        full_prev_sum.extend(prev_sum)
        full_after_sum.extend(after_sum)

    data = torch.tensor(list(zip(full_conf, full_score, full_max_rank, full_prev_sum, full_after_sum, full_token_num))).float()
    target = torch.tensor(correctness).unsqueeze(1).float()
    return data, target


###############################################################################
#                             DATASET CREATION                                #
###############################################################################
def create_dataset(decode_path: str, token_path: str, split: str):
    logdir = os.path.join(decode_path, "logdir")
    word_info = process_data(os.path.join(decode_path, "score_ter/result.txt"), logdir, token_path, 1)
    data, target = get_word_data(decode_path, word_info)

    assert data.shape[1] == len(FEATURE_NAMES), "Mismatch between features and names"
    dataset = NamedTensorDataset(data, target, feature_names=FEATURE_NAMES)

    save_path = os.path.join(CEM_PATH, "dataset", f"{split}_dataset_word{SUFFIX}.pt")
    torch.save(dataset, save_path)
    print(f"{split.capitalize()} dataset saved to {save_path} with {len(data)} entries.")


if __name__ == "__main__":
    # Train dataset
    create_dataset(
        decode_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_finetuning_raw_en_bpe5000/decode_asr_bs20_asr_model_averaged_model_tinv/test_libapt_us_s4",
        token_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt",
        split="train",
    )

    # Uncomment for test set
    # create_dataset(
    #     decode_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_valid.acc.ave/test_libapt_us_subset2500",
    #     token_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt",
    #     split="test",
    # )
