from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import torch


def qwen_eval_relevance(image_path, question, model, tokenizer):

    query_list = [{"image": image_path}]

    query_list.append({"text": question})

    query = tokenizer.from_list_format(query_list)
    outputs = model.chat(
        tokenizer,
        query=query,
        history=None,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
    )

    logits = outputs.scores[0][0]

    probs = (
        torch.nn.functional.softmax(
            torch.FloatTensor(
                [
                    logits[tokenizer("Yes").input_ids[0]],
                    logits[tokenizer("No").input_ids[0]],
                ]
            ),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    return probs[0]


def qwen_chat(image_path, question, model, tokenizer):

    query_list = []
    if image_path:
        for img in image_path.split(","):
            query_list.append({"image": img})

    query_list.append({"text": question})

    query = tokenizer.from_list_format(query_list)
    response, _ = model.chat(tokenizer, query=query, history=None, do_sample=False)

    return response


if __name__ == "__main__":
    model_path = "Qwen/Qwen-VL-Chat"
    adapter_path = (
        "../checkpoints/qwen-vl-chat-2epoch-4batch_size-webqa-reranker-caption-lora"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    mm_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,  # path to the output directory
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    image_path = "../assets/framework.png"
    query = "Image Caption: Centennial Olympic Park splash fountain\nQuestion:\"Are there more than 6 tall lamp posts surrounding the fountain at Centennial Park?\"\nBased on the image and its caption, is the image relevant to the question? Answer 'Yes' or 'No'."
    # ans = qwen_chat(image_path, query, mm_model, tokenizer)
    ans = qwen_eval_relevance(image_path, query, mm_model, tokenizer)
    print(ans)
