import torch
import ipdb
import json
from tqdm import tqdm
import numpy as np
from utils.metrics import webqa_metrics_approx
import argparse


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import llava_chat
from vcd_utils.vcd_sample import evolve_vcd_sampling


############### Inference ###############
def infer(
    model_path,
    image_file,
    question,
    model,
    tokenizer,
    image_processor,
    from_array=False,
    vcd_on=False,
):
    if "webqa" in model_path:
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

    output = llava_chat(
        args, tokenizer, model, image_processor, from_array=from_array, vcd_on=vcd_on
    )

    return output


############### CLIP + Rerank ###############
def baseline_generate(
    val_dataset,
    generator_path,
    tokenizer,
    image_processor,
    generator_model,
    mode,
    vcd_on=False,
):
    acc_scores = {"ALL": [], "Single": [], "Multi": []}

    for guid in tqdm(val_dataset):
        datum = val_dataset[guid]
        question = datum["Q"]
        em_answer = datum["EM"]
        pos_imgs = datum["img_posFacts"]
        qcate = datum["Qcate"]

        pos_source = []

        for item in pos_imgs:
            pos_source.append(str(item["image_id"]))

        IMAGE_PATH = ""
        for i in range(len(pos_source)):
            if mode == "val":
                IMAGE_PATH += "val_image/" + pos_source[i] + ".png"
            elif mode == "dev":
                IMAGE_PATH += "playground/data/train_img/" + pos_source[i] + ".png"
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

        accuracy = webqa_metrics_approx(output, em_answer, qcate)
        acc_scores["ALL"].append(accuracy)

        if len(pos_imgs) == 1:
            acc_scores["Single"].append(accuracy)
        elif len(pos_imgs) > 1:
            acc_scores["Multi"].append(accuracy)

    print("Generation ACC:", np.mean(acc_scores["ALL"]))
    print("Single Img ACC:", np.mean(acc_scores["Single"]))
    print("Multi Imgs ACC:", np.mean(acc_scores["Multi"]))


def evaluate_example():
    with open("answer_sets_webqa/answer_set_base_sft_85_grounding_off.json", "r") as f:
        val_dataset = json.load(f)
    with open("WebQA_val_image_objects.json", "r") as f:
        vanilla_dataset = json.load(f)

    acc_scores = {"ALL": [], "Single": [], "Multi": []}
    for guid, datum in tqdm(val_dataset.items()):
        qcate = vanilla_dataset[guid]["Qcate"]
        prediction = datum["generator_answer"]
        ground_truth = datum["em_answer"]
        # ground_truth = vanilla_dataset[guid]["A"][0]
        num_img = len(datum["gt_images"])
        accuracy = webqa_metrics_approx(prediction, ground_truth, qcate)
        acc_scores["ALL"].append(accuracy)
        if num_img == 1:
            acc_scores["Single"].append(accuracy)
        else:
            acc_scores["Multi"].append(accuracy)

    print("Generation ACC:", np.mean(acc_scores["ALL"]))
    print("Single Img ACC:", np.mean(acc_scores["Single"]))
    print("Multi Imgs ACC:", np.mean(acc_scores["Multi"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="val")
    parser.add_argument("--vcd_on", default=False, action="store_true")
    args = parser.parse_args()

    if args.vcd_on:
        evolve_vcd_sampling()

    # generator_path = "liuhaotian/llava-v1.5-13b"
    # tokenizer, generator_model, image_processor, _ = load_pretrained_model(
    #     model_path=generator_path,
    #     model_base=None,
    #     model_name=get_model_name_from_path(generator_path),
    # )

    # generator_path = (
    #     "checkpoints/llava-v1.5-13b-2epoch-8batch_size-webqa-long-answer-test-lora"
    # )
    generator_path = "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-long-answer-contrastive-distortion-lora-all-ground-truth"
    tokenizer, generator_model, image_processor, _ = load_pretrained_model(
        model_path=generator_path,
        model_base="liuhaotian/llava-v1.5-13b",
        model_name=get_model_name_from_path(generator_path),
    )

    if args.datasets == "val":
        with open("WebQA_val_image_objects.json", "r") as f:
            val_dataset = json.load(f)

    elif args.datasets == "dev":
        with open("WebQA_dev_image.json", "r") as f:
            val_dataset = json.load(f)

    with torch.no_grad():
        baseline_generate(
            val_dataset,
            generator_path,
            tokenizer,
            image_processor,
            generator_model,
            mode=args.datasets,
            vcd_on=args.vcd_on,
        )

    # with torch.no_grad():
    #     answer = infer(
    #         generator_path,
    #         "case_study/1.JPG,case_study/3.JPG",
    #         "Which is better maintained, the carving on the front of the Palace of the Governor in Uxmal or the Bird carving above the doorway in Mexico, Architecture?",
    #         generator_model,
    #         tokenizer,
    #         image_processor,
    #         from_array=False,
    #         vcd_on=False,
    #     )
    #     print(answer)


if __name__ == "__main__":
    # evaluate_example()
    main()