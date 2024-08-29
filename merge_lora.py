import torch
from InternVL.internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer

input_path = (
    "checkpoints/coco/internvl2_2b_1epoch-16batch_size-MSCOCO-one-reranker-caption-lora"
)
output_path = "checkpoints/coco/internvl2_2b_1epoch-16batch_size-MSCOCO-one-reranker-caption-lora-merge"

print("Loading model...")
model = InternVLChatModel.from_pretrained(
    input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
).eval()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)

if model.config.use_backbone_lora:
    model.vision_model.merge_and_unload()
    model.vision_model = model.vision_model.model
    model.config.use_backbone_lora = 0
if model.config.use_llm_lora:
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.model
    model.config.use_llm_lora = 0

print("Saving model...")
model.save_pretrained(output_path)
print("Saving tokenizer...")
tokenizer.save_pretrained(output_path)
print("Done!")
