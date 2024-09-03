from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    CLIPImageProcessor,
)

from llava.mm_utils import get_model_name_from_path
from internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from internvl_chat.internvl.conversation import get_conv_template
from FlagEmbedding.visual.modeling import Visualized_BGE
import clip
import torch

BASE_SERIES = {
    "llava": "liuhaotian/llava-v1.5-13b",
    "owl": "MAGAer13/mplug-owl2-llama2-7b",
    "qwenvl": "Qwen/Qwen-VL-Chat",
    "internvl2-1b": "OpenGVLab/InternVL2-1B",
    "internvl2-2b": "OpenGVLab/InternVL2-2B",
}

RERANKER_SERIES = {
    "webqa": {
        "llava": "checkpoints/web/llava-v1.5-13b-2epoch-16batch_size-webqa-reranker-caption-lora",
        "owl": "checkpoints/web/mplug-owl2-2epoch-8batch_size-webqa-reranker-caption-lora",
        "qwenvl": "checkpoints/web/qwen-vl-chat-2epoch-4batch_size-webqa-reranker-caption-lora-new",
        "internvl2-1b": "checkpoints/web/internvl2_1b_1epoch-16batch_size-webqa-reranker-caption-lora-merge",
        "internvl2-2b": "checkpoints/web/internvl2_2b_1epoch-16batch_size-webqa-reranker-caption-lora-merge",
        "blend": "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-blend-caption-lora",
    },
    "mmqa": {
        "llava": "checkpoints/multimodalqa/llava-v1.5-13b-1epoch-16batch_size-mmqa-reranker-caption-lora",
        "blend": "checkpoints/multimodalqa/llava-v1.5-13b-1epoch-8batch_size-mmqa-blend-caption-lora",
        "internvl2-1b": "checkpoints/multimodalqa/internvl2_1b_1epoch-16batch_size-mmqa-reranker-caption-lora-merge",
        "internvl2-2b": "checkpoints/multimodalqa/internvl2_2b_1epoch-16batch_size-mmqa-reranker-caption-lora-merge",
        "qwenvl": "checkpoints/multimodalqa/qwen-vl-chat-1epoch-4batch_size-mmqa-reranker-caption-lora",
    },
    "flickr30k": {
        "llava": "checkpoints/flickr/llava-v1.5-13b-1epoch-16batch_size-flickr30k-one-reranker-caption-lora",
        "owl": "checkpoints/flickr/mplug-owl2-1epoch-8batch_size-flickr30k-one-reranker-caption-lora",
        "qwenvl": "checkpoints/flickr/qwen-vl-chat-1epoch-4batch_size-flickr30k-one-reranker-caption-lora",
        "internvl2-1b": "checkpoints/flickr/internvl2_1b_1epoch-16batch_size-flickr30k-one-reranker-caption-lora-merge",
        "internvl2-2b": "checkpoints/flickr/internvl2_2b_1epoch-16batch_size-flickr30k-one-reranker-caption-lora-merge",
    },
    "mscoco": {
        "llava": "checkpoints/coco/llava-v1.5-13b-1epoch-16batch_size-MSCOCO-one-reranker-caption-lora",
        "owl": "checkpoints/coco/mplug-owl2-1epoch-8batch_size-MSCOCO-one-reranker-caption-lora",
        "qwenvl": "checkpoints/coco/qwen-vl-chat-1epoch-4batch_size-MSCOCO-one-reranker-caption-lora",
        "internvl2-1b": "checkpoints/coco/internvl2_1b_1epoch-16batch_size-MSCOCO-one-reranker-caption-lora-merge",
        "internvl2-2b": "checkpoints/coco/internvl2_2b_1epoch-16batch_size-MSCOCO-one-reranker-caption-lora-merge",
    },
}

GENERATOR_SERIES = {
    "webqa": {
        "llava": "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-noise-injected-lora",
        "owl": "checkpoints/web/mplug-owl2-2epoch-8batch_size-webqa-noise-injected-lora",
        "qwenvl": "checkpoints/web/qwen-vl-chat-2epoch-2batch_size-webqa-noise-injected-lora-new",
        "internvl2-2b": "checkpoints/web/internvl2_2b_1epoch-8batch_size-webqa-noise-injected-lora-merge",
    },
    "mmqa": {
        "llava": "checkpoints/multimodalqa/llava-v1.5-13b-3epoch-8batch_size-mmqa-noise-injected-lora",
        "internvl2-2b": "checkpoints/multimodalqa/internvl2_2b_1epoch-16batch_size-mmqa-noise-injected-lora-merge",
        "qwenvl": "checkpoints/qwen-vl-chat-1epoch-2batch_size-mmqa-noise-injected-lora",
    },
}


def load_clip(args):
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    if args.clip_type == "clip":
        model, preprocess = clip.load("ViT-L/14@336px", device="cuda", jit=False)
        tokenizer = None
    elif "bge" in args.clip_type:
        bge_model = {
            "bge-base": {
                "name": "BAAI/bge-base-en-v1.5",
                "path": "/data/FinAi_Mapping_Knowledge/chenzhanpeng/RagLLaVA/utils/FlagEmbedding/bge_models/Visualized_base_en_v1.5.pth",
            },
            "bge-m3": {
                "name": "BAAI/bge-m3",
                "path": "/data/FinAi_Mapping_Knowledge/chenzhanpeng/RagLLaVA/utils/FlagEmbedding/bge_models/Visualized_m3.pth",
            },
        }
        model = Visualized_BGE(
            model_name_bge=bge_model[args.clip_type]["name"],
            model_weight=bge_model[args.clip_type]["path"],
        )

        preprocess = None
        tokenizer = None
        model.eval()
    elif "internvl" in args.clip_type.lower():
        model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL-14B-224px",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=0,
        ).eval()

        preprocess = CLIPImageProcessor.from_pretrained("OpenGVLab/InternVL-14B-224px")
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL-14B-224px", use_fast=False, add_eos_token=True
        )
        tokenizer.pad_token_id = 0  # set pad_token_id to 0

    return model, preprocess, tokenizer


def load_reranker(args, dataset, device_map="auto"):
    if args.series not in BASE_SERIES:
        raise ValueError("Invalid model series specified.")

    if args.reranker_model == "base":
        reranker_model_path = BASE_SERIES[args.series]
    else:
        reranker_model_path = RERANKER_SERIES[dataset][args.series]

    print("---------------------------------------")
    print(f"Loading reranker {reranker_model_path}")
    print("---------------------------------------")
    return load_models(args, reranker_model_path, device_map), reranker_model_path


def load_generator(args, dataset, device_map="auto"):
    if args.series not in BASE_SERIES:
        raise ValueError("Invalid model series specified.")

    if args.generator_model == "base":
        generator_model_path = BASE_SERIES[args.series]
    else:
        generator_model_path = GENERATOR_SERIES[dataset][args.series]

    print("---------------------------------------")
    print(f"Loading generator {generator_model_path}")
    print("---------------------------------------")
    return load_models(args, generator_model_path, device_map), generator_model_path


def load_models(args, model_path, device_map="auto"):
    model_series = args.series

    if "llava" in model_series:
        from llava.model.builder import load_pretrained_model

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base="liuhaotian/llava-v1.5-13b" if "lora" in model_path else None,
            model_name=get_model_name_from_path(model_path),
            # use_flash_attn=True,
        )

    elif "owl" in model_series:
        from mplug_owl2.model.builder import load_pretrained_model

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=(
                "MAGAer13/mplug-owl2-llama2-7b" if "lora" in model_path else None
            ),
            model_name=get_model_name_from_path(model_path),
        )

    elif "qwenvl" in model_series:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )

        if "lora" in model_path:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,  # path to the output directory
                device_map=device_map,
                trust_remote_code=True,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", device_map=device_map, trust_remote_code=True
            ).eval()

        image_processor = None

    elif "internvl" in model_series:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_SERIES[model_series], trust_remote_code=True, use_fast=False
        )

        if "lora" in model_path and "merge" not in model_path:
            print("Loading model...")
            model = InternVLChatModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                use_flash_attn=True,
                device_map=device_map,
            ).eval()

            if model.config.use_backbone_lora:
                model.vision_model.merge_and_unload()
                model.vision_model = model.vision_model.model
                model.config.use_backbone_lora = 0
            if model.config.use_llm_lora:
                model.language_model.merge_and_unload()
                model.language_model = model.language_model.model
                model.config.use_llm_lora = 0

            print("Done Merging!")
        else:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()

        image_processor = None
        model.chat = chat.__get__(model, InternVLChatModel)

    return tokenizer, model, image_processor


def chat(
    self,
    tokenizer,
    pixel_values,
    question,
    generation_config,
    history=None,
    return_history=False,
    num_patches_list=None,
    IMG_START_TOKEN="<img>",
    IMG_END_TOKEN="</img>",
    IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    verbose=False,
):

    if history is None and pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id

    template = get_conv_template(self.template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

    history = [] if history is None else history
    for old_question, old_answer in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f"dynamic ViT batch size: {image_bs}")

    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].cuda()
    attention_mask = model_inputs["attention_mask"].cuda()
    generation_config["eos_token_id"] = eos_token_id
    generation_output = self.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config,
    )
    if generation_config.get("return_dict_in_generate", False):
        return generation_output
    else:
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                print(query_to_print, response)
            return response
