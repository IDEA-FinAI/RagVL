import torch
import ipdb
import json
from tqdm import tqdm

import numpy as np

import argparse

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import llava_chat
from mplug_owl2.evaluate.run_mplug_owl2 import owl_chat

from utils.metrics import mmqa_metrics_approx


############### Inference ###############
def infer(
    model_path,
    image_file,
    question,
    model,
    tokenizer,
    image_processor,
    from_array=True,
    vcd_on=False,
):
    if "mmqa" in model_path:
        prompt_template = question
    else:
        prompt_template = f"""Question: {question}\nAnswer the question with less than eight words based on the provided images."""

    args = type(
        "Args",
        (),
        {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt_template,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512,
        },
    )()

    if "llava" in model_path:
        output = llava_chat(
            args,
            tokenizer,
            model,
            image_processor,
            from_array=from_array,
            vcd_on=vcd_on,
        )
    elif "mplug-owl2" in model_path:
        output = owl_chat(args, tokenizer, model, image_processor)

    # elif "qwenvl" in model_path:
    #     output = qwen_chat(args, tokenizer, model, image_processor)

    return output


############### CLIP + Rerank ###############
def baseline_generate(
    val_dataset,
    generator_path,
    tokenizer,
    image_processor,
    generator_model,
    vcd_on=False,
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
            vcd_on=vcd_on,
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
    parser.add_argument("--vcd_on", default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    # if args.vcd_on:
    #     from vcd_utils.vcd_sample import evolve_vcd_sampling
    #     evolve_vcd_sampling()

    # generator_path = "liuhaotian/llava-v1.5-13b"
    generator_path = "MAGAer13/mplug-owl2-llama2-7b"
    # generator_path = (
    #     "checkpoints/llava-v1.5-13b-1epoch-8batch_size-mmqa-noise-injected-lora"
    # )

    if "llava" in generator_path:
        from llava.model.builder import load_pretrained_model

        if "lora" in generator_path:
            tokenizer, generator_model, image_processor, _ = load_pretrained_model(
                model_path=generator_path,
                model_base="liuhaotian/llava-v1.5-13b",
                model_name=get_model_name_from_path(generator_path),
            )
        else:
            tokenizer, generator_model, image_processor, _ = load_pretrained_model(
                model_path=generator_path,
                model_base=None,
                model_name=get_model_name_from_path(generator_path),
            )

    elif "mplug-owl2" in generator_path:
        from mplug_owl2.model.builder import load_pretrained_model

        if "lora" in generator_path:
            tokenizer, generator_model, image_processor, _ = load_pretrained_model(
                model_path=generator_path,
                model_base="MAGAer13/mplug-owl2-llama2-7b",
                model_name=get_model_name_from_path(generator_path),
            )
        else:
            tokenizer, generator_model, image_processor, _ = load_pretrained_model(
                model_path=generator_path,
                model_base=None,
                model_name=get_model_name_from_path(generator_path),
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
            vcd_on=args.vcd_on,
        )


if __name__ == "__main__":
    main()
