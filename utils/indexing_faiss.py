import faiss
import numpy as np
from PIL import Image
import clip
import torch
import json
from tqdm import tqdm

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)
# clip_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)


# ------------- build_index -------------
def build_faiss(val_dataset, device, model):

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
            image_path = "train_img/" + str(image_id) + ".png"
            # image_path = "val_image/" + str(image_id) + ".png"

            image = preprocess(Image.open(image_path)).to(device)
            with torch.no_grad():
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
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


# ------------- text-to-image -------------
def text_to_image(text, model, ind, topk=4):
    text_tokens = clip.tokenize([text], truncate=True)

    text_features = model.encode_text(text_tokens.to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeddings = text_features.cpu().detach().numpy().astype("float32")

    D, I = ind.search(text_embeddings, topk)
    return D, I


def clip_retrieval(val_dataset, ind, index_to_image_id, topk=4):
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
        D, I = text_to_image(question, clip_model, ind, topk)
        for d, j in zip(D[0], I[0]):
            # img_id = index_to_image_id[str(j)]
            img_id = index_to_image_id[j]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    with open("WebQA_train_image.json", "r") as f:
        val_dataset = json.load(f)

    index, index_to_image_id = build_faiss(
        val_dataset, device, clip_model, args.require_caption
    )

    faiss.write_index(index, "WebQA_train_image_large.index")
    with open("WebQA_train_index_to_id.json", "w") as json_file:
        json.dump(index_to_image_id, json_file, indent=4)

    with torch.no_grad():
        hard_examples = clip_retrieval(val_dataset, index, index_to_image_id, args.topk)

    # with open("hard_examples_" + str(args.topk) + ".json", "w") as f:
    #     json.dump(hard_examples, f, indent=4)
