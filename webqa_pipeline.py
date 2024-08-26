import faiss
import numpy as np
import clip
import torch
import ipdb
import json
from tqdm import tqdm
from utils.metrics import webqa_metrics_approx
from utils.indexing_faiss import text_to_image

import argparse

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import llava_chat, llava_eval_relevance
from mplug_owl2.evaluate.run_mplug_owl2 import owl_chat, owl_eval_relevance
from qwenvl.run_qwenvl import qwen_chat, qwen_eval_relevance
from internvl_chat.eval.run_internvl import internvl_chat, internvl_eval_relevance
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from internvl_chat.internvl.model.internvl_chat import InternVLChatModel


def cal_relevance(model_path, image_path, question, model, tokenizer, image_processor):

    if "qwen-vl" in model_path.lower():
        prob = qwen_eval_relevance(image_path, question, model, tokenizer)
    else:
        args = type(
            "Args",
            (),
            {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": question,
                "conv_mode": None,
                "image_file": image_path,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512,
            },
        )()

        if "llava" in model_path:
            prob = llava_eval_relevance(args, tokenizer, model, image_processor)
        elif "mplug-owl2" in model_path:
            prob = owl_eval_relevance(args, tokenizer, model, image_processor)
        elif "internvl" in model_path.lower():
            prob = internvl_eval_relevance(args, tokenizer, model)

    return prob


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


# ------------- CLIP + Rerank -------------


def clip_rerank_generate(
    val_dataset,
    ind,
    index_to_image_id,
    model_path,
    generator_path,
    reranker_model,
    clip_model,
    tokenizer,
    image_processor,
    generator_model,
    save_path,
    filter,
    mode,
    rerank_off,
    use_caption,
    topk=20,
):
    retrieval_correct = 0
    retrieval_num = 0
    retrieval_pos_num = 0

    ### For Retrieval ###
    retrieval_correct_5 = 0
    retrieval_num_5 = 0
    retrieval_correct_10 = 0
    retrieval_num_10 = 0

    acc_scores = {"ALL": [], "Single": [], "Multi": []}

    hard_examples = {}
    probabilities = {"gt": [], "false": []}
    if use_caption:
        if mode == "test":
            with open("datasets/WebQA_caption_test.json", "r") as f:
                captions = json.load(f)
        elif mode == "dev" or mode == "train":
            with open("datasets/WebQA_caption_train_dev.json", "r") as f:
                captions = json.load(f)

    with open(save_path, "w") as f:
        f.write("{\n")
        for guid in tqdm(val_dataset):
            datum = val_dataset[guid]
            question = datum["Q"]
            em_answer = datum["EM"] if "EM" in datum else datum["A"][0]
            pos_imgs = datum["img_posFacts"]
            qcate = datum["Qcate"]

            pos_source = []
            retrieved_imgs = []
            rerank_imgs = {}

            for item in pos_imgs:
                pos_source.append(str(item["image_id"]))
            D, I = text_to_image(question, clip_model, ind, topk)
            for d, j in zip(D[0], I[0]):
                img_id = index_to_image_id[str(j)]
                retrieved_imgs.append(str(img_id))

            if not rerank_off:
                for id in retrieved_imgs:
                    if mode == "test":
                        img_path = "datasets/val_image/" + id + ".png"
                    elif mode == "dev" or mode == "train":
                        img_path = "finetune/tasks/train_img/" + id + ".png"

                    if use_caption:
                        query = (
                            "Image Caption: "
                            + captions[id]
                            + "\nQuestion: "
                            + question
                            + "\nBased on the image and its caption, is the image relevant to the question? Answer 'Yes' or 'No'."
                        )
                    else:
                        query = (
                            "Question: "
                            + question
                            + "\nIs this image relevant to the question? Answer 'Yes' or 'No'."
                        )

                    prob_yes = cal_relevance(
                        model_path,
                        img_path,
                        query,
                        reranker_model,
                        tokenizer,
                        image_processor,
                    )
                    rerank_imgs[id] = float(prob_yes)

                top_sorted_imgs = dict(
                    sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[
                        :2
                    ]
                )

                filtered_imgs = [
                    key for key, val in top_sorted_imgs.items() if val >= filter
                ]

                ### For Retrieval ###
                top_sorted_imgs_5 = dict(
                    sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[
                        :5
                    ]
                )
                top_sorted_imgs_10 = dict(
                    sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[
                        :10
                    ]
                )
                filtered_imgs_5 = [
                    key for key, val in top_sorted_imgs_5.items() if val >= filter
                ]
                filtered_imgs_10 = [
                    key for key, val in top_sorted_imgs_10.items() if val >= filter
                ]

                intersect = set(pos_source).intersection(set(top_sorted_imgs.keys()))
                remaining = set(top_sorted_imgs.keys()).difference(intersect)

                for key in intersect:
                    probabilities["gt"].append(top_sorted_imgs[key])

                for key in remaining:
                    probabilities["false"].append(top_sorted_imgs[key])

            else:
                top_sorted_imgs = retrieved_imgs
                filtered_imgs = retrieved_imgs
                intersect = set(pos_source).intersection(set(retrieved_imgs))
                remaining = set(top_sorted_imgs).difference(intersect)

            if len(intersect) == 0:
                hard_examples[guid] = datum

            retrieval_num += len(filtered_imgs)
            retrieval_pos_num += len(pos_source)
            retrieval_correct += len(set(pos_source).intersection(set(filtered_imgs)))

            ## For Retrieval ###
            retrieval_num_5 += len(filtered_imgs_5)
            retrieval_correct_5 += len(
                set(pos_source).intersection(set(filtered_imgs_5))
            )

            retrieval_num_10 += len(filtered_imgs_10)
            retrieval_correct_10 += len(
                set(pos_source).intersection(set(filtered_imgs_10))
            )

            if generator_model != None:
                IMAGE_PATH = ""
                for i in range(len(filtered_imgs)):
                    if mode == "test":
                        IMAGE_PATH += "datasets/val_image/" + filtered_imgs[i] + ".png"
                    elif mode == "dev":
                        IMAGE_PATH += (
                            "finetune/tasks/train_img/" + filtered_imgs[i] + ".png"
                        )
                    if i != len(filtered_imgs) - 1:
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

                output_json = {
                    "question": question,
                    "generator_answer": output,
                    "em_answer": em_answer,
                    "gt_images": pos_source,
                    "retrieved_images": top_sorted_imgs,
                }
                new_data = json.dumps({guid: output_json})[1:-1]
                f.write(f"    {new_data},\n")

        f.write("}")

    with open("webqa_distribution_prob_" + mode + ".json", "w") as json_file:
        json.dump(probabilities, json_file, indent=4)

    pre = retrieval_correct / retrieval_num
    recall = retrieval_correct / retrieval_pos_num
    f1 = 2 * pre * recall / (pre + recall)

    print("Retrieval pre:", pre)
    print("Retrieval recall:", recall)
    print("Retrieval F1:", f1)

    pre_5 = retrieval_correct_5 / retrieval_num_5
    recall_5 = retrieval_correct_5 / retrieval_pos_num
    f1_5 = 2 * pre_5 * recall_5 / (pre_5 + recall_5)

    print("Retrieval pre_5:", pre_5)
    print("Retrieval recall_5:", recall_5)
    print("Retrieval F1_5:", f1_5)

    pre_10 = retrieval_correct_10 / retrieval_num_10
    recall_10 = retrieval_correct_10 / retrieval_pos_num
    f1_10 = 2 * pre_10 * recall_10 / (pre_10 + recall_10)

    print("Retrieval pre_10:", pre_10)
    print("Retrieval recall_10:", recall_10)
    print("Retrieval F1_10:", f1_10)

    print("Single Img ACC:", np.mean(acc_scores["Single"]))
    print("Multi Imgs ACC:", np.mean(acc_scores["Multi"]))
    print("Generation ACC:", np.mean(acc_scores["ALL"]))

    print("Hard examples count:", len(hard_examples))
    return hard_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker_model", type=str, default="caption_lora")
    parser.add_argument("--generator_model", type=str, default="noise_injected_lora")
    parser.add_argument("--datasets", type=str, default="test")
    parser.add_argument("--filter", type=float, default=0)
    parser.add_argument("--rerank_off", default=False, action="store_true")
    parser.add_argument("--clip_topk", type=int, default=20)
    parser.add_argument("--noise_ratio", type=float, default=0)

    args = parser.parse_args()
    print(args)

    ################### reranker_model ###################

    if args.reranker_model == "base":
        # reranker_model_path = "liuhaotian/llava-v1.5-13b"
        # reranker_model_path = "MAGAer13/mplug-owl2-llama2-7b"
        # reranker_model_path = "Qwen/Qwen-VL-Chat"
        reranker_model_path = "OpenGVLab/InternVL2-2B"

    elif args.reranker_model == "caption_lora":
        # reranker_model_path = "checkpoints/web/llava-v1.5-13b-2epoch-16batch_size-webqa-reranker-caption-lora"
        # reranker_model_path = (
        #     "checkpoints/mplug-owl2-2epoch-8batch_size-webqa-reranker-caption-lora"
        # )
        # reranker_model_path = "checkpoints/web/qwen-vl-chat-2epoch-4batch_size-webqa-reranker-caption-lora-new"

        reranker_model_path = (
            "checkpoints/internvl2_2b_1epoch-16batch_size-webqa-reranker-caption-lora"
        )

    elif args.reranker_model == "blend_caption_lora":
        reranker_model_path = (
            "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-blend-caption-lora"
        )

    if "llava" in reranker_model_path:
        from llava.model.builder import load_pretrained_model

        if "lora" in reranker_model_path:
            tokenizer, reranker_model, image_processor, _ = load_pretrained_model(
                model_path=reranker_model_path,
                model_base="liuhaotian/llava-v1.5-13b",
                model_name=get_model_name_from_path(reranker_model_path),
                # use_flash_attn=True,
            )
        else:
            tokenizer, reranker_model, image_processor, _ = load_pretrained_model(
                model_path=reranker_model_path,
                model_base=None,
                model_name=get_model_name_from_path(reranker_model_path),
            )

    elif "mplug-owl2" in reranker_model_path:
        from mplug_owl2.model.builder import load_pretrained_model

        if "lora" in reranker_model_path:
            tokenizer, reranker_model, image_processor, _ = load_pretrained_model(
                model_path=reranker_model_path,
                model_base="MAGAer13/mplug-owl2-llama2-7b",
                model_name=get_model_name_from_path(reranker_model_path),
            )
        else:
            tokenizer, reranker_model, image_processor, _ = load_pretrained_model(
                model_path=reranker_model_path,
                model_base=None,
                model_name=get_model_name_from_path(reranker_model_path),
            )

    elif "qwen-vl" in reranker_model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )

        if "lora" in reranker_model_path:
            reranker_model = AutoPeftModelForCausalLM.from_pretrained(
                reranker_model_path,  # path to the output directory
                device_map=6,
                trust_remote_code=True,
            ).eval()
        else:
            reranker_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", device_map=2, trust_remote_code=True
            ).eval()

        image_processor = None

    elif "internvl" in reranker_model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            reranker_model_path, trust_remote_code=True, use_fast=False
        )

        if "lora" in reranker_model_path:
            print("Loading model...")
            reranker_model = (
                InternVLChatModel.from_pretrained(
                    reranker_model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

            if reranker_model.config.use_backbone_lora:
                reranker_model.vision_model.merge_and_unload()
                reranker_model.vision_model = reranker_model.vision_model.model
                reranker_model.config.use_backbone_lora = 0
            if reranker_model.config.use_llm_lora:
                reranker_model.language_model.merge_and_unload()
                reranker_model.language_model = reranker_model.language_model.model
                reranker_model.config.use_llm_lora = 0

            print("Done!")
        else:
            reranker_model = (
                AutoModel.from_pretrained(
                    reranker_model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

        image_processor = None

    ################### generator_model ###################
    if args.generator_model == "base":
        # generator_path = "liuhaotian/llava-v1.5-13b"
        # _, generator_model, _, _ = load_pretrained_model(
        #     model_path=generator_path,
        #     model_base=None,
        #     model_name=get_model_name_from_path(generator_path),
        # )

        # generator_path = "Qwen/Qwen-VL-Chat"
        # generator_model = AutoModelForCausalLM.from_pretrained(
        #     generator_path,  # path to the output directory
        #     device_map=7,
        #     trust_remote_code=True,
        # ).eval()

        # generator_path = "MAGAer13/mplug-owl2-llama2-7b"
        # _, generator_model, _, _ = load_pretrained_model(
        #     model_path=generator_path,
        #     model_base=None,
        #     model_name=get_model_name_from_path(generator_path),
        # )

        generator_path = "OpenGVLab/InternVL2-2B"
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

    elif args.generator_model == "blend_lora":
        generator_path = reranker_model_path
        generator_model = reranker_model

    elif args.generator_model == "noise_injected_lora":
        # generator_path = "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-noise-injected-lora"
        # _, generator_model, _, _ = load_pretrained_model(
        #     model_path=generator_path,
        #     model_base="liuhaotian/llava-v1.5-13b",
        #     model_name=get_model_name_from_path(generator_path),
        # )

        # generator_path = (
        #     "checkpoints/qwen-vl-chat-2epoch-2batch_size-webqa-noise-injected-lora-new"
        # )
        # generator_model = AutoPeftModelForCausalLM.from_pretrained(
        #     generator_path,  # path to the output directory
        #     device_map=7,
        #     trust_remote_code=True,
        # ).eval()

        # generator_path = (
        #     "checkpoints/web/mplug-owl2-2epoch-8batch_size-webqa-noise-injected-lora"
        # )
        # _, generator_model, _, _ = load_pretrained_model(
        #     model_path=generator_path,
        #     model_base="MAGAer13/mplug-owl2-llama2-7b",
        #     model_name=get_model_name_from_path(generator_path),
        # )

        generator_path = (
            "checkpoints/internvl2_2b_1epoch-8batch_size-webqa-noise-injected-lora"
        )
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

    elif args.generator_model == "None":
        generator_path = None
        generator_model = None

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    clip_model, preprocess = clip.load("ViT-L/14@336px", device="cuda", jit=False)

    if args.datasets == "test":
        with open("datasets/WebQA_test_image.json", "r") as f:
            val_dataset = json.load(f)

        if args.noise_ratio == 0:
            with open("datasets/WebQA_test_image_index_to_id.json", "r") as f:
                index_to_image_id = json.load(f)
        else:
            with open(
                "datasets/WebQA_test_image_index_to_id_noise"
                + f"{int(args.noise_ratio * 100)}.json",
                "r",
            ) as f:
                index_to_image_id = json.load(f)

        index = faiss.read_index("datasets/faiss_index/WebQA_test_image.index")

    elif args.datasets == "dev":
        with open("datasets/WebQA_dev_image.json", "r") as f:
            val_dataset = json.load(f)

        with open("datasets/WebQA_dev_image_index_to_id.json", "r") as f:
            index_to_image_id = json.load(f)

        index = faiss.read_index("datasets/faiss_index/WebQA_dev_image.index")

    elif args.datasets == "train":
        with open("datasets/WebQA_train_image.json", "r") as f:
            val_dataset = json.load(f)

        with open("datasets/WebQA_train_image_index_to_id.json", "r") as f:
            index_to_image_id = json.load(f)

        index = faiss.read_index("datasets/faiss_index/WebQA_train_image.index")

    save_path = (
        reranker_model_path.split("/")[1]
        + "_webqa_"
        + (
            "_".join(
                [
                    attr
                    for attr in [
                        "answer_set",
                        (
                            f"noise{int(args.noise_ratio * 100)}"
                            if args.noise_ratio != 0
                            else ""
                        ),
                        args.reranker_model,
                        args.generator_model,
                        str(args.filter)[2:],
                        (
                            "clip_top" + str(args.clip_topk)
                            if args.clip_topk != 20
                            else ""
                        ),
                    ]
                    if attr != ""
                ]
            )
            + ".json"
        )
    )

    with torch.no_grad():
        clip_rerank_generate(
            val_dataset,
            index,
            index_to_image_id,
            reranker_model_path,
            generator_path,
            reranker_model,
            clip_model,
            tokenizer,
            image_processor,
            generator_model,
            save_path,
            filter=args.filter,
            mode=args.datasets,
            rerank_off=args.rerank_off,
            use_caption=True if "caption" in args.reranker_model else False,
            topk=args.clip_topk,
        )
