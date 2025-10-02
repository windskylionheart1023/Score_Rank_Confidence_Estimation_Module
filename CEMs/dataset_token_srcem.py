# dataset_creation.py
import os
import re
import json
import torch
import numpy as np
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.asr.CEM.model import NamedTensorDataset


# ---------------- Configuration ----------------
CEM_PATH = "/esat/audioslave/yjia/espnet/espnet/espnet2/asr/CEM"
SUFFIX = "_ours_libapt_s4"
TRANSDUCER = False
NBEST = 1

FEATURE_NAMES = [
    "confidence", "score", "rank",
    "prev_score_sum", "after_score_sum",
    "topk_score_1", "topk_score_2", "topk_score_3", "topk_score_4",
]


# ---------------- Utility Functions ----------------
def process_file_to_dict(input_path: str) -> dict:
    """Parse ESPnet scoring result file into dictionary."""
    with open(input_path, "r") as file:
        lines = file.readlines()

    # keep only part after "Speaker sentences"
    start_idx = next(i for i, line in enumerate(lines) if "Speaker sentences" in line)
    lines = lines[start_idx:]

    data_dict = {}
    for i, line in enumerate(lines):
        if line.startswith("id: ("):
            item_id = line.split()[1].strip("()")
            values = {
                "Scores": lines[i + 1].rstrip(),
                "REF": lines[i + 2][6:].split(),
                "HYP": lines[i + 3][6:].split(),
                "Eval": lines[i + 4].rstrip(),
            }
            data_dict[item_id] = values
    return data_dict


def parse_file_to_dict(filepath: str) -> dict:
    """Parse token_int files into dictionary of ID → token sequence."""
    if "_us" in filepath and "apt" not in filepath:
        id_pattern = re.compile(r"[a-zA-Z0-9]+-common_voice\S*")
    elif "apt" in filepath:
        id_pattern = re.compile(r"en\S*-\d+")
    elif "cgn" in filepath:
        id_pattern = re.compile(r"fv\d+-\d+")
    else:
        id_pattern = re.compile(r"^\d+-\d+-\d+")

    result = {}
    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            match = id_pattern.match(line)
            if not match:
                continue
            id_str = match.group(0)
            content = line[len(id_str):].strip().split()
            content = [int(x) if x.isdigit() else x for x in content]
            result[id_str] = content
    return result


# ---------------- Core Processing ----------------
def process_data(label_file: str, logdir: str, token_path: str, n: int):
    """
    Processes label and score data and returns feature and target tensors.
    """

    # --- Word-level info ---
    wer_result_path = os.path.join(label_file, "score_wer/result.txt")
    wer_data = process_file_to_dict(wer_result_path)
    word_targets = []
    number_of_tokens_per_word = []
    word_correctness_list = []

    # --- Load TER results ---
    ter_file = (
        f"{label_file}/score_ter/result.txt"
        if n == 1 else f"{label_file}/score_ter/{n}best_recog/result.txt"
    )
    label_dict = process_file_to_dict(ter_file)

    # --- Scores and tokens ---
    with open(f"{logdir}/output.1/{n}best_recog/scores_list.json", "r") as f:
        scores_dict = json.load(f)
    hyp_int_dict = parse_file_to_dict(f"{logdir}/output.1/{n}best_recog/token_int")

    tokenidconvertor = TokenIDConverter(token_path)

    # --- Containers ---
    tokens_list, probs_list, confidence_list = [], [], []
    rank_list, score_list = [], []
    prev_score_sum_list, after_score_sum_list = [], []
    correctness_list, topk_scores_list, topk_probs_list = [], [], []

    counter_utt, counter_token = 0, 0

    # --- Loop over utterances ---
    for score_id, score in scores_dict:
        native_conf_list, native_score_list = [], []

        # label ID construction
        if "common_voice" in score_id or "en_us_" in score_id:
            label_id = f"{score_id.split('-')[0]}-{score_id}"
        elif "fv" in score_id:
            label_id = f"{score_id}-spk-{score_id}"
        else:
            parts = score_id.split("-")
            label_id = f"{parts[0]}-{parts[1]}-{score_id}"

        label = label_dict[label_id]["REF"]
        hyp = label_dict[label_id]["HYP"]
        hyp_int = hyp_int_dict[score_id]

        # filter * deletions
        mask = ["*" not in t for t in hyp]
        hyp = [t for t, keep in zip(hyp, mask) if keep]
        label = [t for t, keep in zip(label, mask) if keep]

        if len(hyp) != len(hyp_int) or len(hyp) <= 1:
            counter_utt += 1
            continue

        if "fv" not in score_id:  # normalize case
            label, hyp = [l.upper() for l in label], [h.upper() for h in hyp]

        if len(score) != len(label):
            continue

        label_ids = tokenidconvertor.tokens2ids(label)
        hyp_ids = tokenidconvertor.tokens2ids(hyp)

        utt_token_mask = [1] * len(hyp)
        topk_scores_hyp, topk_probs_hyp = [], []

        for i, hyp_token in enumerate(hyp):
            if hyp_int[i] != hyp_ids[i]:
                counter_token += 1
                utt_token_mask[i] = 0
                continue

            tokens_list.append(hyp_token)
            tensor = torch.full((5000,), float("-inf"))
            for idx_str, val in score[i].items():
                tensor[int(idx_str)] = val

            correctness = float(hyp_token == label[i])
            correctness_list.append(correctness)

            prob_vector = torch.softmax(tensor, dim=0)
            probs_list.append(tensor)

            if TRANSDUCER:
                native_conf = prob_vector[hyp_int[i] - 1].item()
                conf_gap = prob_vector.max().item() - native_conf
                native_score = tensor[hyp_int[i] - 1].item()
                token_rank = list(score[i].keys()).index(str(hyp_int[i] - 1)) + 1
            else:
                native_conf = prob_vector[hyp_int[i]].item()
                conf_gap = prob_vector.max().item() - native_conf
                native_score = tensor[hyp_int[i]].item()
                token_rank = list(score[i].keys()).index(str(hyp_int[i])) + 1

            native_conf_list.append(native_conf)
            native_score_list.append(native_score)
            confidence_list.append(native_conf)
            rank_list.append(token_rank)

            # top-K info
            topk_scores = list(score[i].values())[:4]
            topk_scores_hyp.append(topk_scores)
            topk_probs = torch.topk(prob_vector, 4).values.tolist()
            topk_probs_hyp.append(topk_probs)

            topk_scores_list.append(topk_scores)
            topk_probs_list.append(topk_probs)

        if not native_conf_list:
            counter_utt += 1
            continue

        # accumulate scores
        diff_scores = [native_score_list[0]] + [
            native_score_list[i] - native_score_list[i - 1]
            for i in range(1, len(native_score_list))
        ]
        score_list += diff_scores

        # prefix/suffix sums
        prev_sum = [0] + [sum(diff_scores[:i]) for i in range(1, len(diff_scores))]
        after_sum = [
            sum(diff_scores[i + 1:]) if i < len(diff_scores) - 1 else 0
            for i in range(len(diff_scores))
        ]
        prev_score_sum_list += prev_sum
        after_score_sum_list += after_sum

        # word-level correctness
        hyp_masked = [x for x, m in zip(hyp, utt_token_mask) if m]
        word_boundary = [i for i, tok in enumerate(hyp_masked) if tok.startswith("▁")]
        if word_boundary and word_boundary[-1] != len(hyp_masked):
            word_boundary.append(len(hyp_masked))
        if word_boundary and word_boundary[0] != 0:
            word_boundary.insert(0, 0)

        gt_words = [t for t in wer_data[label_id]["REF"] if "*" not in t]
        pred_words = [t for t in wer_data[label_id]["HYP"] if "*" not in t]

        if len(pred_words) != len(word_boundary) - 1:
            print(f"Warning: Word boundary mismatch in {score_id}, skipping...")
            continue

        utt_tokens_per_word = [
            word_boundary[i + 1] - word_boundary[i]
            for i in range(len(word_boundary) - 1)
        ]
        number_of_tokens_per_word.extend(utt_tokens_per_word)

        for pw, gw in zip(pred_words, gt_words):
            word_correctness_list.append(1 if pw == gw else 0)

    # --- Convert to tensors ---
    data = torch.tensor(
        list(zip(confidence_list, score_list, rank_list, prev_score_sum_list, after_score_sum_list))
    )
    data = torch.cat((data, torch.tensor(topk_scores_list)), dim=1)

    target = torch.tensor(correctness_list).unsqueeze(1)
    word_correctness_tensor = torch.tensor(word_correctness_list).unsqueeze(1).float()

    # Save aux data
    aux_dir = os.path.join(CEM_PATH, "aux")
    os.makedirs(aux_dir, exist_ok=True)
    torch.save(number_of_tokens_per_word, os.path.join(aux_dir, "test_number_of_tokens_per_word_libapt_FT.pt"))
    torch.save(word_correctness_tensor, os.path.join(aux_dir, "test_word_correctness_tensor_libapt_FT.pt"))

    return data, target


# ---------------- Dataset Creation ----------------
def create_dataset(decode_path: str, token_path: str, split: str):
    logdir = os.path.join(decode_path, "logdir")
    nbest_data, nbest_target = [], []

    for n in range(1, NBEST + 1):
        data, target = process_data(decode_path, logdir, token_path, n)
        nbest_data.append(data)
        nbest_target.append(target)

    nbest_data = torch.cat(nbest_data, dim=0)
    nbest_target = torch.cat(nbest_target, dim=0)

    dataset = NamedTensorDataset(nbest_data, nbest_target, feature_names=FEATURE_NAMES)
    save_path = os.path.join(CEM_PATH, "dataset", f"{split}_dataset_token{SUFFIX}.pt")
    torch.save(dataset, save_path)

    print(f"{split.capitalize()} dataset saved to {save_path}")
    print(f"Number of datapoints: {len(nbest_data)}")


if __name__ == "__main__":
    # Train dataset
    create_dataset(
        decode_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_finetuning_raw_en_bpe5000/decode_asr_bs20_asr_model_averaged_model_tinv/test_libapt_us_s4",
        token_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt",
        split="train",
    )

    # Uncomment if you want test set as well:
    # create_dataset(
    #     decode_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/exp/asr_train_asr_conformer_raw_en_bpe5000/decode_asr_bs20_asr_model_valid.acc.ave/train_libapt_us_subset10k",
    #     token_path="/esat/audioslave/yjia/espnet/espnet/egs2/calibration/asr1/data/en_token_list/bpe_unigram5000/tokens.txt",
    #     split="test",
    # )
