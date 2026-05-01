#!/usr/bin/env python3
"""Build a token-level SR-CEM training/test dataset from ESPnet decode outputs.

For every token of the 1-best hypothesis we extract the SR-CEM features
described in Eq. 9 of the paper:

    f_Y(y_t) = [ s_Y(y_t), r(y_t), S_<t, S_>t, Topk(t) ]

i.e. (token score, token rank, preceding-cumulative score, succeeding-cumulative
score, top-K=4 scores). Plus the softmax confidence kept as feature index 0
for diagnostic comparison.

Inputs (per decode directory):
    score_wer/result.txt
    score_ter/result.txt                       (or score_ter/{n}best_recog/result.txt)
    logdir/output.1/{n}best_recog/scores_list.json
    logdir/output.1/{n}best_recog/token_int

The ``scores_list.json`` file is produced by the patched ESPnet code in
this repository (see espnet/espnet/nets/beam_search.py and
espnet/espnet2/bin/asr_inference.py).
"""

import argparse
import json
import os
import re

import torch

from espnet2.asr.CEM.model import NamedTensorDataset
from espnet2.text.token_id_converter import TokenIDConverter


FEATURE_NAMES = [
    "confidence",
    "score",
    "rank",
    "prev_score_sum",
    "after_score_sum",
    "topk_score_1",
    "topk_score_2",
    "topk_score_3",
    "topk_score_4",
]


def process_file_to_dict(input_path: str) -> dict:
    """Parse an ESPnet ``result.txt`` (sclite-style) into ``{utt_id: blocks}``."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    start_idx = next(i for i, line in enumerate(lines) if "Speaker sentences" in line)
    lines = lines[start_idx:]

    data = {}
    for i, line in enumerate(lines):
        if line.startswith("id: ("):
            utt_id = line.split()[1].strip("()")
            data[utt_id] = {
                "Scores": lines[i + 1].rstrip(),
                "REF": lines[i + 2][6:].split(),
                "HYP": lines[i + 3][6:].split(),
                "Eval": lines[i + 4].rstrip(),
            }
    return data


def parse_file_to_dict(filepath: str) -> dict:
    """Parse a ``token_int`` file into ``{utt_id: [token_ids]}``.

    The utterance-id regex is selected from the filepath: LibriSpeech (default),
    CommonVoice US (``_us`` in path), Libri-Adapt (``apt``), or CGN (``cgn``).
    """
    if "_us" in filepath and "apt" not in filepath:
        id_pattern = re.compile(r"[a-zA-Z0-9]+-common_voice\S*")
    elif "apt" in filepath:
        id_pattern = re.compile(r"en\S*-\d+")
    elif "cgn" in filepath:
        id_pattern = re.compile(r"fv\d+-\d+")
    else:
        id_pattern = re.compile(r"^\d+-\d+-\d+")

    result = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = id_pattern.match(line)
            if not match:
                continue
            utt_id = match.group(0)
            content = line[len(utt_id):].strip().split()
            result[utt_id] = [int(x) if x.isdigit() else x for x in content]
    return result


def _label_id_for(score_id: str) -> str:
    """Reconstruct the sclite-style label id from a decoder utterance id."""
    if "common_voice" in score_id or "en_us_" in score_id:
        return f"{score_id.split('-')[0]}-{score_id}"
    if "fv" in score_id:
        return f"{score_id}-spk-{score_id}"
    parts = score_id.split("-")
    return f"{parts[0]}-{parts[1]}-{score_id}"


def process_data(
    decode_dir: str,
    logdir: str,
    token_path: str,
    n: int,
    vocab_size: int,
    transducer: bool,
):
    """Extract per-token feature/target tensors for the n-best=N hypothesis."""

    wer_data = process_file_to_dict(os.path.join(decode_dir, "score_wer/result.txt"))

    ter_file = os.path.join(
        decode_dir,
        "score_ter/result.txt" if n == 1 else f"score_ter/{n}best_recog/result.txt",
    )
    label_dict = process_file_to_dict(ter_file)

    with open(os.path.join(logdir, f"output.1/{n}best_recog/scores_list.json")) as f:
        scores_dict = json.load(f)
    hyp_int_dict = parse_file_to_dict(os.path.join(logdir, f"output.1/{n}best_recog/token_int"))

    token_id_converter = TokenIDConverter(token_path)

    confidence_list, score_list, rank_list = [], [], []
    prev_score_sum_list, after_score_sum_list = [], []
    topk_scores_list = []
    correctness_list = []

    skipped_utts = skipped_tokens = 0

    for score_id, score in scores_dict:
        label_id = _label_id_for(score_id)
        if label_id not in label_dict:
            skipped_utts += 1
            continue

        label = label_dict[label_id]["REF"]
        hyp = label_dict[label_id]["HYP"]
        hyp_int = hyp_int_dict.get(score_id, [])

        # Drop "*" tokens (deletion placeholders).
        mask = ["*" not in t for t in hyp]
        hyp = [t for t, keep in zip(hyp, mask) if keep]
        label = [t for t, keep in zip(label, mask) if keep]

        if len(hyp) != len(hyp_int) or len(hyp) <= 1:
            skipped_utts += 1
            continue

        # Case-fold (CGN labels are already lowercase).
        if "fv" not in score_id:
            label = [tok.upper() for tok in label]
            hyp = [tok.upper() for tok in hyp]

        if len(score) != len(label):
            continue

        hyp_ids = token_id_converter.tokens2ids(hyp)
        utt_token_mask = [1] * len(hyp)

        # Per-token features for this utterance.
        utt_native_scores = []
        utt_topk_scores = []

        for i, hyp_token in enumerate(hyp):
            if hyp_int[i] != hyp_ids[i]:
                # Token-id mismatch (rare): drop this token.
                skipped_tokens += 1
                utt_token_mask[i] = 0
                continue

            tensor = torch.full((vocab_size,), float("-inf"))
            for idx_str, val in score[i].items():
                tensor[int(idx_str)] = val

            prob_vector = torch.softmax(tensor, dim=0)
            target_idx = hyp_int[i] - 1 if transducer else hyp_int[i]

            native_conf = prob_vector[target_idx].item()
            native_score = tensor[target_idx].item()
            token_rank = list(score[i].keys()).index(str(target_idx)) + 1
            topk_scores = list(score[i].values())[:4]

            confidence_list.append(native_conf)
            rank_list.append(token_rank)
            topk_scores_list.append(topk_scores)
            correctness_list.append(float(hyp_token == label[i]))

            utt_native_scores.append(native_score)
            utt_topk_scores.append(topk_scores)

        if not utt_native_scores:
            skipped_utts += 1
            continue

        # Cumulative-score features. ``utt_native_scores`` are cumulative
        # log-probs from beam search, so we difference them to get per-step
        # increments before forming prefix/suffix sums.
        diff_scores = [utt_native_scores[0]] + [
            utt_native_scores[i] - utt_native_scores[i - 1]
            for i in range(1, len(utt_native_scores))
        ]
        prev_sum = [0] + [sum(diff_scores[:i]) for i in range(1, len(diff_scores))]
        after_sum = [
            sum(diff_scores[i + 1:]) if i < len(diff_scores) - 1 else 0
            for i in range(len(diff_scores))
        ]

        score_list.extend(diff_scores)
        prev_score_sum_list.extend(prev_sum)
        after_score_sum_list.extend(after_sum)

        # Sanity check that wer_data is consistent.
        if label_id in wer_data:
            hyp_masked = [x for x, m in zip(hyp, utt_token_mask) if m]
            word_boundary = [i for i, tok in enumerate(hyp_masked) if tok.startswith("▁")]
            if word_boundary and word_boundary[-1] != len(hyp_masked):
                word_boundary.append(len(hyp_masked))
            if word_boundary and word_boundary[0] != 0:
                word_boundary.insert(0, 0)
            pred_words = [t for t in wer_data[label_id]["HYP"] if "*" not in t]
            if len(pred_words) != len(word_boundary) - 1:
                print(f"Warning: word boundary mismatch in {score_id}, skipping word stats.")

    print(
        f"Token-level: kept {len(correctness_list)} tokens; "
        f"skipped {skipped_utts} utterances and {skipped_tokens} mismatched tokens."
    )

    data = torch.tensor(
        list(
            zip(
                confidence_list,
                score_list,
                rank_list,
                prev_score_sum_list,
                after_score_sum_list,
            )
        )
    )
    data = torch.cat((data, torch.tensor(topk_scores_list)), dim=1)
    target = torch.tensor(correctness_list).unsqueeze(1)
    return data, target


def create_dataset(
    decode_path: str,
    token_path: str,
    split: str,
    output_dir: str,
    suffix: str,
    nbest: int,
    vocab_size: int,
    transducer: bool,
):
    """Build the token dataset, save as ``{output_dir}/{split}_dataset_token{suffix}.pt``."""
    logdir = os.path.join(decode_path, "logdir")

    nbest_data, nbest_target = [], []
    for n in range(1, nbest + 1):
        data, target = process_data(decode_path, logdir, token_path, n, vocab_size, transducer)
        nbest_data.append(data)
        nbest_target.append(target)

    nbest_data = torch.cat(nbest_data, dim=0)
    nbest_target = torch.cat(nbest_target, dim=0)

    dataset = NamedTensorDataset(nbest_data, nbest_target, feature_names=FEATURE_NAMES)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{split}_dataset_token{suffix}.pt")
    torch.save(dataset, save_path)

    print(f"Saved {split} token dataset to {save_path} (N={len(nbest_data)})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-path", required=True, help="ESPnet decode directory.")
    parser.add_argument("--token-path", required=True, help="BPE tokens.txt file.")
    parser.add_argument("--output-dir", required=True, help="Where to save the .pt dataset.")
    parser.add_argument("--split", default="train", help="Split tag used in output filename.")
    parser.add_argument("--suffix", default="", help="Optional suffix appended to output filename.")
    parser.add_argument("--nbest", type=int, default=1, help="Use 1..N best hypotheses.")
    parser.add_argument("--vocab-size", type=int, default=5000, help="ASR vocabulary size.")
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
        nbest=args.nbest,
        vocab_size=args.vocab_size,
        transducer=args.transducer,
    )
