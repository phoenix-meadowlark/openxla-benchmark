# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from paxml import trainer_lib
from paxml.tasks.lm.params import lm_cloud
from praxis import base_layer
from praxis import py_utils

from openxla.benchmark.models import model_interfaces

instantiate = base_layer.instantiate
NestedMap = py_utils.NestedMap


class LmCloudSpmd2BFP32(lm_cloud.LmCloudSpmd2B):
  ICI_MESH_SHAPE = [1, 1, 1]
  FPROP_DTYPE = jnp.float32


class LmCloudSpmd2B1g(model_interfaces.InferenceModel):
  """See https://huggingface.co/t5-large for more information."""

  batch_size: int

  def __init__(self, batch_size: int):
    self.batch_size = batch_size

    self.model_config = LmCloudSpmd2BFP32()
    self.model_config.ici_mesh_shape = [1, 1, 1]
    self.model_config.fprop_dtype = jnp.float32
    self.task = instantiate(self.model_config.task())
    self.train_set_p = self.model_config.datasets()[0]
    train_set = instantiate(self.train_set_p)
    train_batch = train_set.get_next()
    self.train_state, _ = trainer_lib.initialize_model_state(
        self.task, jax.random.PRNGKey(123), train_batch)

  def generate_default_inputs(self) -> NestedMap:
    train_set_p = self.train_set_p.clone()
    train_set_p.input.batch_size = self.batch_size
    train_set = instantiate(train_set_p)
    train_batch = train_set.get_next()
    return train_batch

  def preprocess(self, raw_input: Any) -> Any:
    return raw_input

  def forward(self, inputs: NestedMap) -> Tuple[NestedMap, NestedMap]:
    with base_layer.JaxContext(base_layer.JaxContext.HParams()):
      train_state, step_fn_output = (trainer_lib.train_step_single_learner(
          self.task,
          self.train_state,
          jax.random.PRNGKey(123),
          inputs,
      ))
    return step_fn_output

  def postprocess(self, outputs: Any) -> Any:
    return outputs


def create_model(batch_size: int = 1,
                 **_unused_params) -> LmCloudSpmd2B1g:
  """Configure and create a LmCloudSpmd2B model instance.

  Args:
    batch_size: input batch size.
  Returns:
    A LmCloudSpmd2B model.
  """
  return LmCloudSpmd2B1g(batch_size=batch_size)
