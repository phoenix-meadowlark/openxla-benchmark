# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

from openxla.benchmark.devices import gcp_devices
from openxla.benchmark.comparative_suite import utils
from openxla.benchmark.comparative_suite.pt import model_definitions, test_data_definitions

BERT_LARGE_FP32_PT_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP32_PT_384XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_BERT_LARGE_FP32_PT_384XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_BERT_LARGE_FP32_PT_384X1024XF32_BATCHES,
    target_devices=[
        gcp_devices.GCP_A2_HIGHGPU_1G, gcp_devices.GCP_C2_STANDARD_16
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)
BERT_LARGE_FP16_PT_384XI32_CASES = utils.build_batch_benchmark_cases(
    batch_models=model_definitions.BERT_LARGE_FP16_PT_384XI32_BATCHES,
    batch_inputs=test_data_definitions.
    INPUT_DATA_BERT_LARGE_FP16_PT_384XI32_BATCHES,
    batch_expected_outputs=test_data_definitions.
    OUTPUT_DATA_BERT_LARGE_FP16_PT_384X1024XF16_BATCHES,
    target_devices=[gcp_devices.GCP_A2_HIGHGPU_1G],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

ALL_BENCHMARKS = list(
    itertools.chain(
        BERT_LARGE_FP32_PT_384XI32_CASES.values(),
        BERT_LARGE_FP16_PT_384XI32_CASES.values(),
    ))
