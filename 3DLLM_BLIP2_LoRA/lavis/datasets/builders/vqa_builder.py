"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.threedvqa_datasets import ThreeDVQADataset, ThreeDVQAEvalDataset
from lavis.datasets.datasets.threedvqa_datasets import ScanQADataset, ScanQAEvalDataset
from lavis.datasets.datasets.threedvqa_datasets import SQADataset, SQAEvalDataset
from lavis.datasets.datasets.threedvqa_datasets import PretrainDataset, PretrainDatasetEvalDataset

@registry.register_builder("3d_vqa")
class ThreeDVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDVQADataset
    eval_dataset_cls = ThreeDVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dvqa/defaults.yaml"}

@registry.register_builder("scanqa")
class ScanVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ScanQADataset
    eval_dataset_cls = ScanQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dvqa/scanqa.yaml"}

@registry.register_builder("sqa")
class ScanVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = SQADataset
    eval_dataset_cls = SQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dvqa/sqa.yaml"}

@registry.register_builder("pretrain")
class ScanVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainDataset
    eval_dataset_cls = PretrainDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dvqa/pretrain.yaml"}
    