import faiss
import numpy as np
from PIL import Image
import clip
import torch
import json
from tqdm import tqdm
import argparse
import pandas as pd

from FlagEmbedding.visual.modeling import Visualized_BGE
import ipdb
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer
from model_series import load_clip

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------- build_index -------------
def build_faiss(val_dataset, device, model, clip_type="clip", preprocess=None):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for i in tqdm(val_dataset):
        datum = val_dataset[i]
        pos_imgs = datum["img_posFacts"]

        for j in range(len(pos_imgs)):
            image_id = pos_imgs[j]["image_id"]
            if image_id in index_to_image_id.values():
                continue
            # image_path = "../finetune/tasks/train_img/" + str(image_id) + ".png"
            image_path = "../datasets/val_image/" + str(image_id) + ".png"

            with torch.no_grad():
                if clip_type == "clip":
                    image = preprocess(Image.open(image_path)).to(device)
                    image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
                elif "bge" in clip_type:
                    image_embeddings = model.encode(image=image_path)
                else:
                    pixel_values = preprocess(
                        images=Image.open(image_path).convert("RGB"),
                        return_tensors="pt",
                    ).pixel_values
                    pixel_values = pixel_values.to(torch.bfloat16).to(device)
                    image_embeddings = model.encode_image(
                        pixel_values, mode=clip_type
                    ).to(torch.float)

            combined_embedding = image_embeddings
            normalized_embedding = combined_embedding / combined_embedding.norm(
                dim=-1, keepdim=True
            )
            embeddings.append(normalized_embedding.cpu().numpy())

            index_to_image_id[count] = image_id
            count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # Euclidean
    # index = faiss.IndexFlatL2(embeddings.shape[1])

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_image_id


def build_faiss_mmqa(
    val_dataset, metadata, device, model, clip_type="clip", preprocess=None
):

    embeddings = []
    index_to_image_id = {}
    count = 0
    for datum in tqdm(val_dataset):
        pos_img = datum["supporting_context"][0]
        image_id = pos_img["doc_id"]
        if image_id in index_to_image_id.values():
            continue
        image_path = "../finetune/tasks/MMQA_imgs/" + metadata[image_id]["path"]

        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(Image.open(image_path)).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            else:
                pixel_values = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_image_id


def build_faiss_flickr30k(
    val_dataset, device, model, clip_type="clip", preprocess=None
):
    embeddings = []
    index_to_image_id = {}
    count = 0
    images = val_dataset["image"]
    for img in tqdm(images[::5]):
        image_path = "../finetune/tasks/flickr30k/Images/" + img
        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(Image.open(image_path)).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            else:
                pixel_values = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = img[:-4]
        count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_image_id


def build_faiss_coco(
    original_data, val_dataset, device, model, clip_type="clip", preprocess=None
):
    embeddings = []
    index_to_image_id = {}
    count = 0
    image_ids = val_dataset["ids"]
    image_files = {}
    for img in original_data["images"]:
        image_files[str(img["id"])] = img["file_name"]

    for img in tqdm(image_ids[::5]):
        image_path = "../finetune/tasks/MSCOCO/val2014/" + image_files[img]
        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(Image.open(image_path)).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            else:
                pixel_values = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = img
        count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_image_id


# ------------- text-to-image -------------
def text_to_image(text, model, ind, topk=4, clip_type="clip", tokenizer=None):
    if clip_type == "clip":
        text_tokens = clip.tokenize([text], truncate=True)
        text_features = model.encode_text(text_tokens.to(device))
    elif "bge" in clip_type:
        text_features = model.encode(text=text)
    else:
        prefix = "summarize:"
        text = prefix + text
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            max_length=80,
            truncation=True,
            padding="max_length",
        ).input_ids.to(device)

        text_features = model.encode_text(input_ids).to(torch.float)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeddings = text_features.cpu().detach().numpy().astype("float32")

    D, I = ind.search(text_embeddings, topk)
    return D, I


def clip_retrieval(val_dataset, ind, index_to_image_id, model, args, tokenizer=None):
    total_correct = 0
    total_sc_num = 0
    total_re_num = 0
    hard_examples = {}
    for i in tqdm(val_dataset):
        pos_source = []
        retrieved_imgs = []
        datum = val_dataset[i]
        question = datum["Q"]
        pos_imgs = datum["img_posFacts"]
        if len(pos_imgs) == 0:
            continue
        for item in pos_imgs:
            pos_source.append(item["image_id"])
        D, I = text_to_image(
            question,
            model,
            ind,
            args.topk,
            clip_type=args.clip_type,
            tokenizer=tokenizer,
        )
        for d, j in zip(D[0], I[0]):
            ## 从json文件中load index_to_image_id的话要用这一行
            img_id = index_to_image_id[str(j)]

            # img_id = index_to_image_id[j]
            retrieved_imgs.append(img_id)

        intersect = set(pos_source).intersection(set(retrieved_imgs))
        if len(intersect) == 0:
            hard_examples[i] = datum

        total_sc_num += len(retrieved_imgs)
        total_re_num += len(pos_source)
        total_correct += len(list(intersect))

    pre = total_correct / total_sc_num
    recall = total_correct / total_re_num
    f1 = 2 * pre * recall / (pre + recall)
    print("Re pre:", pre)
    print("Re recall:", recall)
    print("Re F1:", f1)
    print("Hard examples count:", len(hard_examples))
    return hard_examples


def clip_retrieval_mmqa(
    val_dataset, ind, index_to_image_id, model, args, tokenizer=None
):
    total_correct = 0
    total_sc_num = 0
    total_re_num = 0
    hard_examples = []
    for datum in tqdm(val_dataset):
        pos_source = []
        retrieved_imgs = []
        question = datum["question"]
        pos_imgs = datum["supporting_context"]
        for item in pos_imgs:
            pos_source.append(item["doc_id"])

        D, I = text_to_image(
            question,
            model,
            ind,
            args.topk,
            clip_type=args.clip_type,
            tokenizer=tokenizer,
        )
        for d, j in zip(D[0], I[0]):
            img_id = index_to_image_id[str(j)]

            # img_id = index_to_image_id[j]
            retrieved_imgs.append(img_id)

        intersect = set(pos_source).intersection(set(retrieved_imgs))
        if len(intersect) == 0:
            hard_examples.append(datum)

        total_sc_num += len(retrieved_imgs)
        total_re_num += len(pos_source)
        total_correct += len(list(intersect))

    pre = total_correct / total_sc_num
    recall = total_correct / total_re_num
    f1 = 2 * pre * recall / (pre + recall)
    print("Re pre:", pre)
    print("Re recall:", recall)
    print("Re F1:", f1)
    print("Hard examples count:", len(hard_examples))
    return hard_examples


def clip_retrieval_flickr30k(
    val_dataset, ind, index_to_image_id, model, args, tokenizer=None
):
    total_correct = 0
    total_sc_num = 0
    total_re_num = 0

    captions = val_dataset["caption"]
    images = val_dataset["image"]
    for i in tqdm(range(len(captions))):
        cap = captions[i]
        pos_source = [images[i][:-4]]
        retrieved_imgs = []

        D, I = text_to_image(
            cap, model, ind, args.topk, clip_type=args.clip_type, tokenizer=tokenizer
        )
        for d, j in zip(D[0], I[0]):
            ## 从json文件中load index_to_image_id的话要用这一行
            img_id = index_to_image_id[str(j)]
            # img_id = index_to_image_id[j]

            retrieved_imgs.append(img_id)

        intersect = set(pos_source).intersection(set(retrieved_imgs))
        total_sc_num += len(retrieved_imgs)
        total_re_num += len(pos_source)
        total_correct += len(list(intersect))

    pre = total_correct / total_sc_num
    recall = total_correct / total_re_num
    f1 = 2 * pre * recall / (pre + recall)
    print("Re pre:", pre)
    print("Re recall:", recall)
    print("Re F1:", f1)


def clip_retrieval_coco(
    val_dataset, ind, index_to_image_id, model, args, tokenizer=None
):
    total_correct = 0
    total_sc_num = 0
    total_re_num = 0

    captions = val_dataset["captions"]
    image_ids = val_dataset["ids"]
    for i in tqdm(range(len(captions))):
        cap = captions[i]
        pos_source = [image_ids[i]]
        retrieved_imgs = []

        D, I = text_to_image(
            cap, model, ind, args.topk, clip_type=args.clip_type, tokenizer=tokenizer
        )
        for d, j in zip(D[0], I[0]):
            ## 从json文件中load index_to_image_id的话要用这一行
            img_id = index_to_image_id[str(j)]
            # img_id = index_to_image_id[j]

            retrieved_imgs.append(img_id)

        intersect = set(pos_source).intersection(set(retrieved_imgs))
        total_sc_num += len(retrieved_imgs)
        total_re_num += len(pos_source)
        total_correct += len(list(intersect))

    pre = total_correct / total_sc_num
    recall = total_correct / total_re_num
    f1 = 2 * pre * recall / (pre + recall)
    print("Re pre:", pre)
    print("Re recall:", recall)
    print("Re F1:", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="WebQA")
    parser.add_argument("--clip_type", type=str, default="clip")
    args = parser.parse_args()

    model, preprocess, tokenizer = load_clip(args)

    if args.datasets == "WebQA":
        with open("../datasets/WebQA_test_image.json", "r") as f:
            val_dataset = json.load(f)
        # index, index_to_image_id = build_faiss(
        #     val_dataset,
        #     device,
        #     model,
        #     clip_type=args.clip_type,
        #     preprocess=preprocess,
        # )

    elif args.datasets == "MMQA":

        with open("../datasets/MMQA_test_image.json", "r") as f:
            val_dataset = json.load(f)
        with open("../datasets/MMQA_ImageQ_metadata.json", "r") as f:
            metadata = json.load(f)

        # index, index_to_image_id = build_faiss_mmqa(
        #     val_dataset,
        #     metadata,
        #     device,
        #     model,
        #     clip_type=args.clip_type,
        #     preprocess=preprocess,
        # )

    elif args.datasets == "flickr30k":
        val_dataset = pd.read_csv("../datasets/flickr30k_test_karpathy.txt")
        # index, index_to_image_id = build_faiss_flickr30k(
        #     val_dataset,
        #     device,
        #     model,
        #     clip_type=args.clip_type,
        #     preprocess=preprocess,
        # )

    elif args.datasets == "coco":
        ids = []
        captions = []
        with open("../datasets/coco_test_ids.txt", "r") as f:
            for line in f:
                ids.append(line.strip())

        with open("../datasets/coco_test_caps.txt", "r") as f:
            for line in f:
                captions.append(line.strip())

        val_dataset = {"ids": ids, "captions": captions}

        with open("../datasets/coco_test_karpathy.json", "r") as f:
            original_data = json.load(f)

        # index, index_to_image_id = build_faiss_coco(
        #     original_data,
        #     val_dataset,
        #     device,
        #     model,
        #     clip_type=args.clip_type,
        #     preprocess=preprocess,
        # )

    # faiss.write_index(
    #     index,
    #     "../datasets/faiss_index/"
    #     + args.datasets
    #     + "_test_image_"
    #     + args.clip_type
    #     + ".index",
    # )

    index = faiss.read_index(
        "../datasets/faiss_index/"
        + args.datasets
        + "_test_image_"
        + args.clip_type
        + ".index"
    )
    with open(
        "../datasets/" + args.datasets + "_test_image_index_to_id.json", "r"
    ) as f:
        index_to_image_id = json.load(f)

    with torch.no_grad():
        if args.datasets == "WebQA":
            hard_examples = clip_retrieval(
                val_dataset, index, index_to_image_id, model, args, tokenizer
            )
        elif args.datasets == "MMQA":
            clip_retrieval_mmqa(
                val_dataset, index, index_to_image_id, model, args, tokenizer
            )
        elif args.datasets == "flickr30k":
            clip_retrieval_flickr30k(
                val_dataset, index, index_to_image_id, model, args, tokenizer
            )
        elif args.datasets == "coco":
            clip_retrieval_coco(
                val_dataset, index, index_to_image_id, model, args, tokenizer
            )
