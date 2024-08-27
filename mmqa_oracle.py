import torch
import ipdb
import json
from tqdm import tqdm
import numpy as np

from utils.metrics import mmqa_metrics_approx
from utils.model_series import load_generator
from utils.utils import infer
import argparse


############### CLIP + Rerank ###############
def baseline_generate(
    val_dataset,
    generator_path,
    tokenizer,
    image_processor,
    generator_model,
):
    acc_scores = {"ALL": []}

    with open("datasets/MMQA_ImageQ_metadata.json", "r") as f:
        metadata = json.load(f)

    for datum in tqdm(val_dataset):
        qid = datum["qid"]
        question = datum["question"]
        answer = datum["answers"][0]["answer"]
        pos_imgs = datum["supporting_context"]

        pos_source = []

        for item in pos_imgs:
            pos_source.append(item["doc_id"])

        IMAGE_PATH = ""
        for i in range(len(pos_source)):
            IMAGE_PATH += "finetune/tasks/MMQA_imgs/" + metadata[pos_source[i]]["path"]
            if i != len(pos_source) - 1:
                IMAGE_PATH += ","

        output = infer(
            generator_path,
            IMAGE_PATH,
            question,
            generator_model,
            tokenizer,
            image_processor,
            from_array=False,
        )

        if "how many" in question.lower():
            qcate = "number"
        else:
            qcate = "normal"

        accuracy = mmqa_metrics_approx(output, answer, qcate)
        acc_scores["ALL"].append(accuracy)

    print("Generation ACC:", np.mean(acc_scores["ALL"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="test")
    parser.add_argument("--generator_model", type=str, default="noise_injected_lora")
    parser.add_argument("--series", type=str, default="llava")
    args = parser.parse_args()
    print(args)

    tokenizer, generator_model, image_processor, generator_path = load_generator(
        args, "mmqa"
    )

    if args.datasets == "test":
        with open("datasets/MMQA_test_ImageQ.json", "r") as f:
            val_dataset = json.load(f)

    elif args.datasets == "dev":
        with open("datasets/MMQA_dev_ImageQ.json", "r") as f:
            val_dataset = json.load(f)

    with torch.no_grad():
        baseline_generate(
            val_dataset,
            generator_path,
            tokenizer,
            image_processor,
            generator_model,
        )


if __name__ == "__main__":
    main()
