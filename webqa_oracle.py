import torch
import ipdb
import json
from tqdm import tqdm
import numpy as np
from utils.metrics import webqa_metrics_approx
import argparse


from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import llava_chat
from mplug_owl2.evaluate.run_mplug_owl2 import owl_chat
from qwenvl.run_qwenvl import qwen_chat
from internvl_chat.eval.run_internvl import internvl_chat, internvl_eval_relevance

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import AutoPeftModelForCausalLM
from internvl_chat.internvl.model.internvl_chat import InternVLChatModel


############### Inference ###############
def infer(
    model_path,
    image_file,
    question,
    model,
    tokenizer,
    image_processor,
    from_array=False,
):
    if "webqa" in model_path:
        prompt_template = question
    else:
        prompt_template = f"""Question: {question}\nAnswer the question with less than eight words based on the provided images."""

    if "qwen-vl" in model_path.lower():
        output = qwen_chat(image_file, prompt_template, model, tokenizer)
    else:
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
            )
        elif "mplug-owl2" in model_path:
            output = owl_chat(args, tokenizer, model, image_processor)
        elif "internvl" in model_path.lower():
            output = internvl_chat(args, tokenizer, model)

    return output


############### Noise Injection ###############
def inject_noise(val_dataset, noise_ratio=0):
    id_to_image = {}
    idx = []
    for _, datum in val_dataset.items():
        pos_imgs = datum["img_posFacts"]
        for img in pos_imgs:
            if not (str(img["image_id"]) in id_to_image):
                idx.append(str(img["image_id"]))
                id_to_image[str(img["image_id"])] = str(img["image_id"])

    if noise_ratio:
        origin_length = len(idx)
        np.random.shuffle(idx)
        noise_length = int(noise_ratio * origin_length)

        shuffle_index = [id_to_image[i] for i in idx[:noise_length]]
        np.random.shuffle(shuffle_index)

        for i, shuffle_value in enumerate(shuffle_index):
            id_to_image[idx[i]] = shuffle_value

    count = 0
    noise = 0
    for key, value in id_to_image.items():
        if key == value:
            count += 1
        else:
            noise += 1

    cal_noise_ratio = noise / (count + noise)
    print(
        "=> the noise_ratio is {} and the cal_noise_ratio is {}".format(
            noise_ratio, cal_noise_ratio
        )
    )
    with open(
        "WebQA_test_image_id_to_image_noise" + f"{int(noise_ratio*100)}.json", "w"
    ) as f:
        json.dump(id_to_image, f, indent=4)

    return id_to_image


############### CLIP + Rerank ###############
def baseline_generate(
    val_dataset,
    generator_path,
    tokenizer,
    image_processor,
    generator_model,
    mode,
    noise_ratio=0,
):
    acc_scores = {"ALL": [], "Single": [], "Multi": []}

    if noise_ratio != 0:
        # id_to_image = inject_noise(val_dataset, noise_ratio)

        with open(
            "WebQA_test_image_id_to_image_noise" + f"{int(noise_ratio*100)}.json", "r"
        ) as f:
            id_to_image = json.load(f)

    for guid in tqdm(val_dataset):
        datum = val_dataset[guid]
        question = datum["Q"]
        em_answer = datum["EM"]
        pos_imgs = datum["img_posFacts"]
        qcate = datum["Qcate"]

        pos_source = []

        for item in pos_imgs:
            if noise_ratio != 0:
                pos_source.append(id_to_image[str(item["image_id"])])
            else:
                pos_source.append(str(item["image_id"]))

        IMAGE_PATH = ""
        for i in range(len(pos_source)):
            if mode == "test":
                IMAGE_PATH += "datasets/val_image/" + pos_source[i] + ".png"
            elif mode == "dev":
                IMAGE_PATH += "finetune/tasks/train_img/" + pos_source[i] + ".png"
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

        accuracy = webqa_metrics_approx(output, em_answer, qcate)
        acc_scores["ALL"].append(accuracy)

        if len(pos_imgs) == 1:
            acc_scores["Single"].append(accuracy)
        elif len(pos_imgs) > 1:
            acc_scores["Multi"].append(accuracy)

    print("Single Img ACC:", np.mean(acc_scores["Single"]))
    print("Multi Imgs ACC:", np.mean(acc_scores["Multi"]))
    print("Generation ACC:", np.mean(acc_scores["ALL"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="test")
    parser.add_argument("--noise_ratio", type=float, default=0)
    args = parser.parse_args()
    print(args)

    generator_path = "OpenGVLab/InternVL2-2B"
    # generator_path = "liuhaotian/llava-v1.5-13b"
    # generator_path = "MAGAer13/mplug-owl2-llama2-7b"
    # generator_path = "Qwen/Qwen-VL-Chat"

    # generator_path = (
    #     "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-vanilla-lora"
    # )
    # generator_path = (
    #     "checkpoints/web/mplug-owl2-2epoch-8batch_size-webqa-noise-injected-lora"
    # )
    # generator_path = (
    #     "checkpoints/internvl2_2b_1epoch-8batch_size-webqa-noise-injected-lora"
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

    elif "qwen-vl" in generator_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )

        if "lora" in generator_path:
            generator_model = AutoPeftModelForCausalLM.from_pretrained(
                generator_path,  # path to the output directory
                device_map="auto",
                trust_remote_code=True,
            ).eval()
        else:
            generator_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True
            ).eval()

        image_processor = None

    elif "internvl" in generator_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            generator_path, trust_remote_code=True, use_fast=False
        )

        if "lora" in generator_path:
            print("Loading model...")
            generator_model = (
                InternVLChatModel.from_pretrained(
                    generator_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

            if generator_model.config.use_backbone_lora:
                generator_model.vision_model.merge_and_unload()
                generator_model.vision_model = generator_model.vision_model.model
                generator_model.config.use_backbone_lora = 0
            if generator_model.config.use_llm_lora:
                generator_model.language_model.merge_and_unload()
                generator_model.language_model = generator_model.language_model.model
                generator_model.config.use_llm_lora = 0

            print("Done!")
        else:
            generator_model = (
                AutoModel.from_pretrained(
                    generator_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

        image_processor = None

    if args.datasets == "test":
        with open("datasets/WebQA_test_image.json", "r") as f:
            val_dataset = json.load(f)

    elif args.datasets == "dev":
        with open("datasets/WebQA_dev_image.json", "r") as f:
            val_dataset = json.load(f)

    with torch.no_grad():
        baseline_generate(
            val_dataset,
            generator_path,
            tokenizer,
            image_processor,
            generator_model,
            mode=args.datasets,
            noise_ratio=args.noise_ratio,
        )


if __name__ == "__main__":
    main()
