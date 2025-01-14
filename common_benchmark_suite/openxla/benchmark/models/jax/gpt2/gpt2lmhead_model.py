# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from transformers import AutoTokenizer, GPT2Tokenizer, FlaxGPT2LMHeadModel
from typing import Any, List, Tuple

from openxla.benchmark.models import model_interfaces


class GPT2LMHead(model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/gpt2 for more information."""

  batch_size: int
  seq_len: int
  model: FlaxGPT2LMHeadModel
  model_name: str
  tokenizer: GPT2Tokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(
      self,
      batch_size: int,
      seq_len: int,
      model_name: str,
  ):
    self.model = FlaxGPT2LMHeadModel.from_pretrained(model_name)
    self.model_name = model_name
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenization_kwargs = {
        "pad_to_multiple_of": self.seq_len,
        "padding": True,
        "return_tensors": "jax",
    }

  def generate_default_inputs(self) -> str:
    return "a photo of a cat"

  def preprocess(self, input_text: str) -> Tuple[Any, Any]:
    batch_input_text = [input_text] * self.batch_size
    inputs = self.tokenizer(text=batch_input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"], inputs["attention_mask"])

  def forward(self, input_ids: Any, attention_mask: Any) -> Any:
    return self.model(input_ids, attention_mask).logits

  def postprocess(self, output: Any) -> List[str]:
    return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def create_model(batch_size: int = 1,
                 seq_len: int = 512,
                 model_name: str = "gpt2",
                 **_unused_params) -> GPT2LMHead:
  """Configure and create a JAX GPT2LMHead model instance.

  Args:
    batch_size: input batch size.
    seq_len: input sequence length.
    model_name: The name of the GPT2 variant to use. Supported variants include:
      gpt2.
  Returns:
    A JAX GPT2LMHead model.
  """
  return GPT2LMHead(batch_size=batch_size,
                    seq_len=seq_len,
                    model_name=model_name)
