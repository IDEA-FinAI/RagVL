import torch
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.mm_utils import (
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def owl_chat(args, tokenizer, model, image_processor):

    query = args.query

    if len(args.image_file) != 0:
        image_files = image_parser(args)
        images = load_images(image_files)
        images_tensor = process_images(images, image_processor, model.config)

        if type(images_tensor) is list:
            for i in range(len(images_tensor)):
                images_tensor[i] = images_tensor[i].to(
                    model.device, dtype=torch.float16
                )
        else:
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)

        for i in range(len(image_files)):
            query = DEFAULT_IMAGE_TOKEN + "\n" + query
    else:
        images_tensor = None

    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            # forward_func="vanilla",
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

    return outputs


def owl_eval_relevance(args, tokenizer, model, image_processor):

    query = args.query

    if args.image_file != "":
        image_files = image_parser(args)
        images = load_images(image_files)
        images_tensor = process_images(images, image_processor, model.config)

        if type(images_tensor) is list:
            for i in range(len(images_tensor)):
                images_tensor[i] = images_tensor[i].to(
                    model.device, dtype=torch.float16
                )
        else:
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)

        query = DEFAULT_IMAGE_TOKEN + "\n" + query
    else:
        images_tensor = None

    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        generation_output = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_scores=True,
            # forward_func="vanilla",
        )

    logits = generation_output.scores[0][0]

    probs = (
        torch.nn.functional.softmax(
            torch.FloatTensor(
                [
                    logits[tokenizer("Yes").input_ids[1]],
                    logits[tokenizer("No").input_ids[1]],
                ]
            ),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    return probs[0]
