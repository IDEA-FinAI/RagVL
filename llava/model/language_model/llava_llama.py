#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import ipdb

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ...constants import IGNORE_INDEX
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from vcd_utils.vcd_add_noise import add_diffusion_noise


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.avg_layer = torch.nn.AvgPool1d(
            3, stride=1, padding=1, count_include_pad=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cd_inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_cd: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        forward_func: Optional[str] = "vanilla",
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        ################### vcd ###################
        # if images_cd is not None:
        #     if cd_inputs_embeds is None or inputs_embeds is None:
        #         (
        #             input_ids,
        #             position_ids,
        #             attention_mask,
        #             past_key_values,
        #             cd_inputs_embeds,
        #             labels,
        #         ) = self.prepare_inputs_labels_for_multimodal(
        #             input_ids,
        #             position_ids,
        #             attention_mask,
        #             past_key_values,
        #             labels,
        #             images_cd,
        #             image_sizes,
        #         )

        #     return super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=cd_inputs_embeds,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )

        # else:
        #     if inputs_embeds is None:
        #         (
        #             input_ids,
        #             position_ids,
        #             attention_mask,
        #             past_key_values,
        #             inputs_embeds,
        #             labels,
        #         ) = self.prepare_inputs_labels_for_multimodal(
        #             input_ids,
        #             position_ids,
        #             attention_mask,
        #             past_key_values,
        #             labels,
        #             images,
        #             image_sizes,
        #         )

        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        if forward_func == "vanilla":
            ################### vanilla forward ###################
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes,
                )

            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
            ################## distortion-image cd-weighted ###################
            if inputs_embeds is None:
                images_tensor_cd = [
                    add_diffusion_noise(img_tensor, 500) for img_tensor in images
                ]
                if all(x.shape == images_tensor_cd[0].shape for x in images_tensor_cd):
                    images_tensor_cd = torch.stack(images_tensor_cd, dim=0).to(
                        images[0].device, dtype=images[0].dtype
                    )

                (
                    (
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        inputs_embeds,
                        labels,
                    ),
                    (
                        _,
                        _,
                        _,
                        _,
                        cd_inputs_embeds,
                        _,
                    ),
                ) = (
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        labels,
                        images,
                        image_sizes,
                    ),
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        labels,
                        images_tensor_cd,
                        image_sizes,
                    ),
                )

            return self.cd_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                cd_inputs_embeds=cd_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    def cd_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cd_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.training and labels is not None:
            with torch.no_grad():
                cd_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=cd_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                cd_hidden_states = cd_outputs[0]
                if self.pretraining_tp > 1:
                    lm_head_slices = self.lm_head.weight.split(
                        self.vocab_size // self.pretraining_tp, dim=0
                    )
                    cd_logits = [
                        F.linear(cd_hidden_states, lm_head_slices[i])
                        for i in range(self.pretraining_tp)
                    ]
                    cd_logits = torch.cat(cd_logits, dim=-1)
                else:
                    cd_logits = self.lm_head(cd_hidden_states)
                cd_logits = cd_logits.float()

                del cd_hidden_states, cd_outputs

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)

        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()
        del hidden_states

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            with torch.no_grad():
                shift_logits_clone = shift_logits.clone()
                shift_cd_logits = cd_logits[..., :-1, :].contiguous()
                shift_cd_logits = shift_cd_logits.view(-1, self.config.vocab_size)
                del cd_logits

                #### VCD Style ####
                # cutoff = (
                #     torch.log(torch.tensor(0.5))
                #     + shift_logits_clone.max(dim=-1, keepdim=True).values
                # )

                # diffs = shift_logits_clone - shift_cd_logits
                # sub = diffs.masked_fill(shift_logits_clone < cutoff, -float("inf"))

                sub = shift_logits_clone - shift_cd_logits

                sub = torch.clamp(sub, min=1, max=5)
                sub = self.avg_layer(sub[:, 0].unsqueeze(dim=0).unsqueeze(dim=0))
                # sub = torch.exp(sub)
                # sub = sub + 1e-6
                # sub = self.avg_layer(sub[:, 0].unsqueeze(dim=0).unsqueeze(dim=0))
                sub = sub[0, 0]

                del shift_cd_logits, shift_logits_clone

            loss = loss_fct(shift_logits, shift_labels) * sub
            loss = loss.sum() / (sub[shift_labels != IGNORE_INDEX].sum() + 1e-6)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        images_cd = kwargs.get("images_cd", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images_cd is not None:
            (_, cd_position_ids, cd_attention_mask, _, cd_inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images_cd,
                    image_sizes=image_sizes,
                )
            )
            kwargs["cd_inputs_embeds"] = cd_inputs_embeds

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes

        inputs["forward_func"] = kwargs.pop("forward_func", None)

        return inputs

    def prepare_inputs_for_generation_cd(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        images_cd = kwargs.pop("images_cd", None)
        image_sizes = kwargs.pop("image_sizes", None)
        cd_inputs_embeds = kwargs.pop("cd_inputs_embeds", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images_cd is not None:
            inputs["images_cd"] = images_cd
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if cd_inputs_embeds is not None:
            inputs["cd_inputs_embeds"] = cd_inputs_embeds
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
