from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import llava_chat, llava_eval_relevance
from mplug_owl2.evaluate.run_mplug_owl2 import owl_chat, owl_eval_relevance
from qwenvl.run_qwenvl import qwen_chat, qwen_eval_relevance
from InternVL.internvl_chat.eval.run_internvl import (
    internvl_chat,
    internvl_eval_relevance,
)


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
    if any(dataset_type in model_path for dataset_type in ("webqa", "mmqa")):
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
