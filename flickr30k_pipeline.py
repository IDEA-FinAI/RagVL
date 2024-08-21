import faiss
import clip
import torch
import ipdb
import json
from tqdm import tqdm
from utils.indexing_faiss import text_to_image
import pandas as pd
import argparse

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import llava_eval_relevance
from mplug_owl2.evaluate.run_mplug_owl2 import owl_eval_relevance
from qwenvl.run_qwenvl import qwen_eval_relevance
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    return prob


# ------------- CLIP + Rerank -------------


def clip_rerank_generate(
    val_dataset,
    ind,
    index_to_image_id,
    model_path,
    reranker_model,
    clip_model,
    tokenizer,
    image_processor,
    filter,
    rerank_off,
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

    captions = val_dataset["caption"]
    images = val_dataset["image"]
    for i in tqdm(range(len(captions))):
        cap = captions[i]
        pos_source = [images[i][:-4]]
        retrieved_imgs = []
        rerank_imgs = {}

        D, I = text_to_image(cap, clip_model, ind, topk)
        for d, j in zip(D[0], I[0]):
            img_id = index_to_image_id[str(j)]
            retrieved_imgs.append(str(img_id))

        if not rerank_off:
            for id in retrieved_imgs:
                img_path = "finetune/tasks/flickr30k/Images/" + id + ".jpg"

                query = (
                    "Image Caption: "
                    + cap
                    + "\nIs the image relevant to the caption? Answer 'Yes' or 'No'."
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
                sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[:1]
            )

            filtered_imgs = [
                key for key, val in top_sorted_imgs.items() if val >= filter
            ]

            ### For Retrieval ###
            top_sorted_imgs_5 = dict(
                sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[:5]
            )
            top_sorted_imgs_10 = dict(
                sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[:10]
            )
            filtered_imgs_5 = [
                key for key, val in top_sorted_imgs_5.items() if val >= filter
            ]
            filtered_imgs_10 = [
                key for key, val in top_sorted_imgs_10.items() if val >= filter
            ]

        else:
            top_sorted_imgs = retrieved_imgs
            filtered_imgs = retrieved_imgs

        retrieval_num += len(filtered_imgs)
        retrieval_pos_num += len(pos_source)
        retrieval_correct += len(set(pos_source).intersection(set(filtered_imgs)))

        ## For Retrieval ###
        retrieval_num_5 += len(filtered_imgs_5)
        retrieval_correct_5 += len(set(pos_source).intersection(set(filtered_imgs_5)))

        retrieval_num_10 += len(filtered_imgs_10)
        retrieval_correct_10 += len(set(pos_source).intersection(set(filtered_imgs_10)))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker_model", type=str, default="caption_lora")
    parser.add_argument("--filter", type=float, default=0)
    parser.add_argument("--rerank_off", default=False, action="store_true")
    parser.add_argument("--clip_topk", type=int, default=20)

    args = parser.parse_args()
    print(args)

    ################### reranker_model ###################
    if args.reranker_model == "base":
        # reranker_model_path = "liuhaotian/llava-v1.5-13b"
        # reranker_model_path = "MAGAer13/mplug-owl2-llama2-7b"
        reranker_model_path = "Qwen/Qwen-VL-Chat"

    elif args.reranker_model == "caption_lora":
        # reranker_model_path = "checkpoints/flickr/llava-v1.5-13b-1epoch-16batch_size-flickr30k-reranker-caption-lora"
        reranker_model_path = "checkpoints/llava-v1.5-13b-2epoch-16batch_size-flickr30k-reranker-caption-lora"

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
                device_map=0,
                trust_remote_code=True,
            ).eval()
        else:
            reranker_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", device_map=0, trust_remote_code=True
            ).eval()

        image_processor = None

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    clip_model, preprocess = clip.load("ViT-L/14@336px", device="cuda", jit=False)

    val_dataset = pd.read_csv("datasets/flickr30k_test_karpathy.txt")

    with open("datasets/flickr30k_test_image_index_to_id.json", "r") as f:
        index_to_image_id = json.load(f)

    index = faiss.read_index("datasets/faiss_index/flickr30k_test_image.index")

    with torch.no_grad():
        clip_rerank_generate(
            val_dataset,
            index,
            index_to_image_id,
            reranker_model_path,
            reranker_model,
            clip_model,
            tokenizer,
            image_processor,
            filter=args.filter,
            rerank_off=args.rerank_off,
            topk=args.clip_topk,
        )
