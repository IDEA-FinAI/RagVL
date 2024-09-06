import faiss
import numpy as np
import torch
import ipdb
import json
from tqdm import tqdm
from utils.metrics import mmqa_metrics_approx
from utils.indexing_faiss import text_to_image
from utils.model_series import load_generator, load_reranker, load_clip
from utils.utils import infer, cal_relevance
import argparse


# ------------- CLIP + Rerank -------------
def clip_rerank_generate(
    val_dataset,
    ind,
    index_to_image_id,
    reranker_model_path,
    generator_path,
    reranker_model,
    tokenizer,
    image_processor,
    generator_model,
    save_path,
    clip_model,
    clip_tokenizer,
    clip_type,
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

    acc_scores = {"ALL": []}

    hard_examples = {}
    probabilities = {"gt": [], "false": []}
    with open("datasets/MMQA_ImageQ_metadata.json", "r") as f:
        metadata = json.load(f)

    with open(save_path, "w") as f:
        f.write("{\n")
        for datum in tqdm(val_dataset):
            qid = datum["qid"]
            question = datum["question"]
            answer = datum["answers"][0]["answer"]
            pos_imgs = datum["supporting_context"]

            pos_source = []
            retrieved_imgs = []
            rerank_imgs = {}

            for item in pos_imgs:
                pos_source.append(item["doc_id"])

            D, I = text_to_image(
                question, clip_model, ind, topk, clip_type, clip_tokenizer
            )
            for d, j in zip(D[0], I[0]):
                img_id = index_to_image_id[str(j)]
                retrieved_imgs.append(img_id)

            if not rerank_off:
                for id in retrieved_imgs:
                    img_path = "finetune/tasks/MMQA_imgs/" + metadata[id]["path"]
                    img_caption = metadata[id]["caption"]

                    if use_caption:
                        query = (
                            "Image Caption: "
                            + img_caption
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
                        reranker_model_path,
                        img_path,
                        query,
                        reranker_model,
                        tokenizer,
                        image_processor,
                    )
                    rerank_imgs[id] = float(prob_yes)

                ####### MMQA取Top-1, WebQA取Top-2 #######
                top_sorted_imgs = dict(
                    sorted(rerank_imgs.items(), key=lambda item: item[1], reverse=True)[
                        :1
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
                hard_examples[qid] = datum

            retrieval_num += len(filtered_imgs)
            retrieval_pos_num += len(pos_source)
            retrieval_correct += len(set(pos_source).intersection(set(filtered_imgs)))

            ### For Retrieval ###
            if not rerank_off:
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
                    IMAGE_PATH += (
                        "finetune/tasks/MMQA_imgs/" + metadata[filtered_imgs[i]]["path"]
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

                if "how many" in question.lower():
                    qcate = "number"
                else:
                    qcate = "normal"
                accuracy = mmqa_metrics_approx(output, answer, qcate)
                acc_scores["ALL"].append(accuracy)

                output_json = {
                    "question": question,
                    "generator_answer": output,
                    "answer": answer,
                    "gt_images": pos_source,
                    "retrieved_images": top_sorted_imgs,
                }
                new_data = json.dumps({qid: output_json})[1:-1]
                f.write(f"    {new_data},\n")  # 写入缩进的键值对

        f.write("}")

    pre = retrieval_correct / retrieval_num
    recall = retrieval_correct / retrieval_pos_num
    f1 = 2 * pre * recall / (pre + recall)

    print("Retrieval pre:", pre)
    print("Retrieval recall:", recall)
    print("Retrieval F1:", f1)

    if not rerank_off:

        with open(
            "logs/mmqa/"
            + reranker_model_path.split("/")[-1]
            + "_mmqa_distribution_prob_"
            + mode
            + ".json",
            "w",
        ) as json_file:
            json.dump(probabilities, json_file, indent=4)

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

    print("Generation ACC:", np.mean(acc_scores["ALL"]))

    print("Hard examples count:", len(hard_examples))
    return hard_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_type", type=str, default="clip")
    parser.add_argument("--reranker_model", type=str, default="caption_lora")
    parser.add_argument("--generator_model", type=str, default="noise_injected_lora")
    parser.add_argument("--series", type=str, default="llava")
    parser.add_argument("--datasets", type=str, default="test")
    parser.add_argument("--filter", type=float, default=0)
    parser.add_argument("--rerank_off", default=False, action="store_true")
    parser.add_argument("--clip_topk", type=int, default=20)

    args = parser.parse_args()
    print(args)

    clip_model, _, clip_tokenizer = load_clip(args)

    if not args.rerank_off:
        (tokenizer, reranker_model, image_processor), reranker_model_path = (
            load_reranker(args, "mmqa")
        )
    else:
        reranker_model = None
        reranker_model_path = "off/rerank_off"

    if args.generator_model == "blend_lora":
        generator_path = reranker_model_path
        generator_model = reranker_model
    elif args.generator_model == "None":
        generator_model = None
        generator_path = None
    else:
        (tokenizer, generator_model, image_processor), generator_path = load_generator(
            args, "mmqa"
        )

    with open("datasets/MMQA_" + args.datasets + "_image.json", "r") as f:
        val_dataset = json.load(f)

    with open("datasets/MMQA_" + args.datasets + "_image_index_to_id.json", "r") as f:
        index_to_image_id = json.load(f)

    index = faiss.read_index(
        "datasets/faiss_index/MMQA_"
        + args.datasets
        + "_image_"
        + args.clip_type
        + ".index"
    )

    save_path = (
        "logs/mmqa/"
        + reranker_model_path.split("/")[1]
        + "_mmqa_"
        + (
            "_".join(
                [
                    attr
                    for attr in [
                        "answer_set",
                        args.reranker_model,
                        args.generator_model,
                        str(args.filter)[2:],
                        (
                            "clip_top" + str(args.clip_topk)
                            if args.clip_topk != 20
                            else ""
                        ),
                        args.datasets,
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
            tokenizer,
            image_processor,
            generator_model,
            save_path,
            clip_model,
            clip_tokenizer,
            clip_type=args.clip_type,
            filter=args.filter,
            mode=args.datasets,
            rerank_off=args.rerank_off,
            use_caption=True if "caption" in args.reranker_model else False,
            topk=args.clip_topk,
        )

    print(args)
