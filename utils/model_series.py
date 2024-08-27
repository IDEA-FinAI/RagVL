from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from InternVL.internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from llava.mm_utils import get_model_name_from_path
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
        "internvl2-1b": "checkpoints/internvl2_1b_1epoch-16batch_size-webqa-reranker-caption-lora",
        "internvl2-2b": "checkpoints/internvl2_2b_1epoch-16batch_size-webqa-reranker-caption-lora",
        "blend": "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-blend-caption-lora",
    },
    "mmqa": {
        "llava": "checkpoints/multimodalqa/llava-v1.5-13b-1epoch-16batch_size-mmqa-reranker-caption-lora",
        "blend": "checkpoints/multimodalqa/llava-v1.5-13b-1epoch-8batch_size-mmqa-blend-caption-lora",
    },
    "flickr30k": {
        "llava": "checkpoints/flickr/llava-v1.5-13b-1epoch-16batch_size-flickr30k-one-reranker-caption-lora",
        "owl": "checkpoints/flickr/mplug-owl2-1epoch-8batch_size-flickr30k-one-reranker-caption-lora",
        "qwenvl": "checkpoints/flickr/qwen-vl-chat-1epoch-4batch_size-flickr30k-one-reranker-caption-lora",
    },
    "mscoco": {
        "llava": "checkpoints/coco/llava-v1.5-13b-1epoch-16batch_size-MSCOCO-one-reranker-caption-lora",
        "owl": "checkpoints/coco/mplug-owl2-1epoch-8batch_size-MSCOCO-one-reranker-caption-lora",
        "qwenvl": "checkpoints/coco/qwen-vl-chat-1epoch-4batch_size-MSCOCO-one-reranker-caption-lora",
    },
}

GENERATOR_SERIES = {
    "webqa": {
        "llava": "checkpoints/web/llava-v1.5-13b-2epoch-8batch_size-webqa-noise-injected-lora",
        "owl": "checkpoints/web/mplug-owl2-2epoch-8batch_size-webqa-noise-injected-lora",
        "qwenvl": "checkpoints/web/qwen-vl-chat-2epoch-2batch_size-webqa-noise-injected-lora-new",
        "internvl2-2b": "checkpoints/internvl2_2b_1epoch-8batch_size-webqa-noise-injected-lora",
    },
    "mmqa": {
        "llava": "checkpoints/multimodalqa/llava-v1.5-13b-1epoch-8batch_size-mmqa-noise-injected-lora"
    },
}


def load_reranker(args, dataset):
    if args.series not in BASE_SERIES:
        raise ValueError("Invalid model series specified.")

    if args.reranker_model == "base":
        reranker_model_path = BASE_SERIES[args.series]
    else:
        reranker_model_path = RERANKER_SERIES[dataset][args.series]

    return load_models(reranker_model_path), reranker_model_path


def load_generator(args, dataset):
    if args.series not in BASE_SERIES:
        raise ValueError("Invalid model series specified.")

    if args.generator_model == "base":
        generator_model_path = BASE_SERIES[args.series]
    else:
        generator_model_path = GENERATOR_SERIES[dataset][args.series]

    return load_models(generator_model_path), generator_model_path


def load_models(model_path):
    model_path = model_path.lower()

    if "llava" in model_path:
        from llava.model.builder import load_pretrained_model

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base="liuhaotian/llava-v1.5-13b" if "lora" in model_path else None,
            model_name=get_model_name_from_path(model_path),
            # use_flash_attn=True,
        )

    elif "mplug-owl2" in model_path:
        from mplug_owl2.model.builder import load_pretrained_model

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=(
                "MAGAer13/mplug-owl2-llama2-7b" if "lora" in model_path else None
            ),
            model_name=get_model_name_from_path(model_path),
        )

    elif "qwen-vl" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )

        if "lora" in model_path:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,  # path to the output directory
                device_map=6,
                trust_remote_code=True,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", device_map=2, trust_remote_code=True
            ).eval()

        image_processor = None

    elif "internvl" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        if "lora" in model_path:
            print("Loading model...")
            model = (
                InternVLChatModel.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

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
            model = (
                AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

        image_processor = None

    return tokenizer, model, image_processor
