# RagLLaVA
This is the official repo for paper: ["MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training"](https://arxiv.org/pdf/2407.21439).

![image](https://github.com/IDEA-FinAI/RagLLaVA/tree/main/assets/framework.png)

## Updates
- [2024-08-05]: Codes of RagLLaVA released.
- [2024-07-31]: Paper of RagLLaVA online.

## Getting Started
### Environment Setup
The required libraries for running RagLLaVA can be found in `requirements.txt`.

### Data Preparation
Before running RagLLaVA, please:

1. Download from [Google Drive](https://drive.google.com/drive/folders/1wY18Vbrb8yDbFSg1Te-FQIs84AYYh48Z?usp=drive_link) for **datasets** and **checkpoints**. 

2. Download from [WebQA](https://github.com/WebQnA/WebQA) and [MultimodalQA](https://github.com/allenai/multimodalqa) for **image files**.

3. Unzip the file. Place the `checkpoints/` and `datasets/` into `RagLLaVA/`.

4. Place the `tasks/` into `RagLLaVA/finetune/`.

5. Place the `MMQA_imgs/` and `train_img/` into `RagLLaVA/finetune/tasks/`.

6. Place the `val_image/` into `RagLLaVA/datasets/` .


## Evaluation
To eval RagLLaVA on WebQA / MultimodalQA, you can employ the following command:

```
python webqa_pipeline.py \  # same arguments on mmqa_pipeline.py
--reranker_model caption_lora \ # select the reranker
--generator_model noise_injected_lora \ # select the generator
--filter 0 \ # select the adaptive threshold
--clip_topk 20 \ # we first retrieve 20 candidates by default
```

To eval the oracle settings on WebQA / MultimodalQA, you can employ the following command:

```
python webqa_oracle.py \  # same arguments on mmqa_oracle.py
```

## Citation
If you interested or inspired by this work, you can cite us by:
```sh
@article{chen2024mllm,
  title={MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training},
  author={Chen, Zhanpeng and Xu, Chengjin and Qi, Yiyan and Guo, Jian},
  journal={arXiv preprint arXiv:2407.21439},
  year={2024}
}
```

## Related Projects
- [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
- [VCD](https://github.com/DAMO-NLP-SG/VCD): Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
- [CAL](https://github.com/foundation-multimodal-models/CAL): Prioritizing Visual Correlation by Contrastive Alignment
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL): A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond