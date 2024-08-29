import torch
import ipdb
import json
from tqdm import tqdm
import numpy as np
from utils.metrics import webqa_metrics_approx
from utils.model_series import load_generator
from utils.utils import infer
import argparse


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
    parser.add_argument("--generator_model", type=str, default="noise_injected_lora")
    parser.add_argument("--series", type=str, default="llava")
    parser.add_argument("--noise_ratio", type=float, default=0)
    args = parser.parse_args()
    print(args)

    (tokenizer, generator_model, image_processor), generator_path = load_generator(
        args, "webqa"
    )

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

    print(args)


if __name__ == "__main__":
    main()
