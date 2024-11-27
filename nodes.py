"""该代码文件定义了一系列与Flux模型训练相关的类和函数，用于实现模型的初始化、训练循环、验证、保存、提取LoRA权重等功能。这些类和函数主要用于ComfyUI框架中进行深度学习模型的训练。"""

import os
import torch
from torchvision import transforms

import folder_paths
import comfy.model_management as mm
import comfy.utils
import toml
import json
import time
import shutil
import shlex

from pathlib import Path
script_directory = os.path.dirname(os.path.abspath(__file__))

from .flux_train_network_comfy import FluxNetworkTrainer
from .library import flux_train_utils as  flux_train_utils
from .flux_train_comfy import FluxTrainer
from .flux_train_comfy import setup_parser as train_setup_parser
from .library.device_utils import init_ipex
init_ipex()

from .library import train_util
from .train_network import setup_parser as train_network_setup_parser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxTrainModelSelect:
    """- **功能**: 选择用于训练的模型文件。
* 输入
  - `transformer`: UNET模型文件路径。
  - `vae`: VAE模型文件路径。
  - `clip_l`: CLIP模型文件路径。
  - `t5`: T5模型文件路径。
  - `lora_path`: 可选的预训练LoRA权重文件路径。
- **输出**: 包含所有选定模型路径的字典。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "transformer": (folder_paths.get_filename_list("unet"), ),
                    "vae": (folder_paths.get_filename_list("vae"), ),
                    "clip_l": (folder_paths.get_filename_list("clip"), ),
                    "t5": (folder_paths.get_filename_list("clip"), ),
                },
                "optional": {
                    "lora_path": ("STRING",{"multiline": True, "forceInput": True, "default": "", "tooltip": "pre-trained LoRA path to load (network_weights)"}),
                }
        }

    RETURN_TYPES = ("TRAIN_FLUX_MODELS",)
    RETURN_NAMES = ("flux_models",)
    FUNCTION = "loadmodel"
    CATEGORY = "FluxTrainer"

    def loadmodel(self, transformer, vae, clip_l, t5, lora_path=""):
        """输入: transformer, vae, clip_l, t5, lora_path
        内部逻辑: 获取每个组件的完整路径，并将它们存储在一个字典中。
        输出: 包含模型路径的字典。"""
        
        transformer_path = folder_paths.get_full_path("unet", transformer)
        vae_path = folder_paths.get_full_path("vae", vae)
        clip_path = folder_paths.get_full_path("clip", clip_l)
        t5_path = folder_paths.get_full_path("clip", t5)

        flux_models = {
            "transformer": transformer_path,
            "vae": vae_path,
            "clip_l": clip_path,
            "t5": t5_path,
            "lora_path": lora_path
        }
        
        return (flux_models,)

class TrainDatasetGeneralConfig:
    """主要功能
        - **功能**: 配置数据集的一般参数。
    - **输入**:
    - `color_aug`: 是否启用弱颜色增强。
    当开启颜色增强时，图像的色调每次都会随机改变。LoRA 从中学到的预期是颜色色调会有轻微的范围。
    - `flip_aug`: 是否启用水平翻转增强。
    如果此选项开启，图片将随机水平翻转。它可以学习左右角度，这在您想学习对称人物和物体时很有用。
    - `shuffle_caption`: 是否在训练时打乱标签顺序。
    如果训练图像有标题，大多数标题都是以逗号分隔的单词形式编写的，例如“黑猫，吃，坐着”。“随机打乱标题”选项会每次随机改变这些逗号分隔的单词的顺序。
    标题中的词语通常越靠近开头，其权重就越大。因此，如果词序固定，反向词语可能学得不好，正向词语可能与训练图像产生意想不到的关联。希望可以通过每次加载图像时重新排序词语来纠正这种偏差。
    此选项在标题不是以逗号分隔而是写成句子时没有意义。
    在这里，“单词”指的是由逗号分隔的文本片段。无论分隔的文本包含多少个单词，它都算作“一个单词”。
    在“黑猫，吃，坐”的情况下，“黑猫”是一个词。
    - `caption_dropout_rate`: 标签丢弃率，范围在0.0到1.0之间。
    它类似于每 n 个 epoch 进行 Dropout，但你可以在不使用标题的情况下学习“无标题的图像”，在整个学习过程中的一定比例中不使用标题。
    这里可以设置不带标题的图片百分比。0 是“学习时始终使用标题”的设置，1 是“学习时从不使用标题”的设置。
    它是随机的，哪些图像被学习为“无标题图像”。
    例如，如果读取 20 张图像，每张图像读取 50 次，并且仅进行 1 个 epoch 的 LoRA 学习，则图像学习的总次数为 20 张图像 x 50 次 x 1 个 epoch = 1000 次。此时，如果设置标题丢失率（Rate of caption dropout）为 0.1，则 1000 次 x 0.1 = 100 次将作为“无标题图像”进行学习。
    - `alpha_mask`: 是否使用alpha通道作为掩码进行训练。
    
    - **输出**: 包含配置信息的JSON对象。"""
    queue_counter = 0
    @classmethod
    def IS_CHANGED(s, reset_on_queue=False, **kwargs):
        if reset_on_queue:
            s.queue_counter += 1
        print(f"queue_counter: {s.queue_counter}")
        return s.queue_counter
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "color_aug": ("BOOLEAN",{"default": False, "tooltip": "enable weak color augmentation"}),
            "flip_aug": ("BOOLEAN",{"default": False, "tooltip": "enable horizontal flip augmentation"}),
            "shuffle_caption": ("BOOLEAN",{"default": False, "tooltip": "shuffle caption"}),
            "caption_dropout_rate": ("FLOAT",{"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "tag dropout rate"}),
            "alpha_mask": ("BOOLEAN",{"default": False, "tooltip": "use alpha channel as mask for training"}),
            },
            "optional": {
                "reset_on_queue": ("BOOLEAN",{"default": False, "tooltip": "Force refresh of everything for cleaner queueing"}),
                "caption_extension": ("STRING",{"default": ".txt", "tooltip": "extension for caption files"}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("dataset_general",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, shuffle_caption, caption_dropout_rate, color_aug, flip_aug, alpha_mask, reset_on_queue=False, caption_extension=".txt"):
        """输入: shuffle_caption, caption_dropout_rate, color_aug, flip_aug, alpha_mask, reset_on_queue, caption_extension

内部逻辑: 创建一个包含数据集一般设置的JSON对象。

输出: 包含数据集设置的JSON对象。

使用时机: 在配置数据集时使用。"""
        
        dataset = {
           "general": {
                "shuffle_caption": shuffle_caption,
                "caption_extension": caption_extension,
                "keep_tokens_separator": "|||",
                "caption_dropout_rate": caption_dropout_rate,
                "color_aug": color_aug,
                "flip_aug": flip_aug,
           },
           "datasets": []
        }
        dataset_json = json.dumps(dataset, indent=2)
        #print(dataset_json)
        dataset_config = {
            "datasets": dataset_json,
            "alpha_mask": alpha_mask
        }
        return (dataset_config,)

class TrainDatasetRegularization:
    """- **功能**: 创建正则化子集配置。
  - **输入**:
  - `dataset_path`: 数据集路径，根目录是'ComfyUI'或'ComfyUI_windows_portable'（对于Windows便携版）。
  - `class_tokens`: 类别标记词（触发词），如果指定，将在每个标签前添加此标记词；如果没有标签，则单独使用此标记词。
  - `num_repeats`: 数据集重复次数以增加一个epoch中的数据量。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "dataset_path": ("STRING",{"multiline": True, "default": "", "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "class_tokens": ("STRING",{"multiline": True, "default": "", "tooltip": "aka trigger word, if specified, will be added to the start of each caption, if no captions exist, will be used on it's own"}),
            "num_repeats": ("INT", {"default": 1, "min": 1, "tooltip": "number of times to repeat dataset for an epoch"}),
            },
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("subset",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, dataset_path, class_tokens, num_repeats):
        
        reg_subset = {
                    "image_dir": dataset_path,
                    "class_tokens": class_tokens,
                    "num_repeats": num_repeats,
                    "is_reg": True
                }
       
        return reg_subset,
    
class TrainDatasetAdd:  
    """主要功能
添加数据集配置，包括分辨率、批量大小、数据集路径等。

**参数说明:**
- **dataset_config**: 已存在的数据集配置 JSON 对象。
- **width**, **height**: 图像的基本分辨率宽度和高度（最小值为64）。
- **batch_size**: 批处理大小（最小值为1）。
- **dataset_path**, **class_tokens**, etc.: 数据集路径、类别标记等配置信息。"""
    def __init__(self):
        self.previous_dataset_signature = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "dataset_config": ("JSON",),
            "width": ("INT",{"min": 64, "default": 1024, "tooltip": "base resolution width"}),
            "height": ("INT",{"min": 64, "default": 1024, "tooltip": "base resolution height"}),
            "batch_size": ("INT",{"min": 1, "default": 2, "tooltip": "Higher batch size uses more memory and generalizes the training more"}),
            "dataset_path": ("STRING",{"multiline": True, "default": "", "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "class_tokens": ("STRING",{"multiline": True, "default": "", "tooltip": "aka trigger word, if specified, will be added to the start of each caption, if no captions exist, will be used on it's own"}),
            "enable_bucket": ("BOOLEAN",{"default": True, "tooltip": "enable buckets for multi aspect ratio training"}),
            "bucket_no_upscale": ("BOOLEAN",{"default": False, "tooltip": "don't allow upscaling when bucketing"}),
            "num_repeats": ("INT", {"default": 1, "min": 1, "tooltip": "number of times to repeat dataset for an epoch"}),
            "min_bucket_reso": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 8, "tooltip": "min bucket resolution"}),
            "max_bucket_reso": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "max bucket resolution"}),
            },
            "optional": {
                 "regularization": ("JSON", {"tooltip": "reg data dir"}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, dataset_config, dataset_path, class_tokens, width, height, batch_size, num_repeats, enable_bucket,  
                  bucket_no_upscale, min_bucket_reso, max_bucket_reso, regularization=None):
        
        new_dataset = {
            "resolution": (width, height),
            "batch_size": batch_size,
            "enable_bucket": enable_bucket,
            "bucket_no_upscale": bucket_no_upscale,
            "min_bucket_reso": min_bucket_reso,
            "max_bucket_reso": max_bucket_reso,
            "subsets": [
                {
                    "image_dir": dataset_path,
                    "class_tokens": class_tokens,
                    "num_repeats": num_repeats
                }
            ]
        }
        if regularization is not None:
            new_dataset["subsets"].append(regularization)

        # Generate a signature for the new dataset
        new_dataset_signature = self.generate_signature(new_dataset)

        # Load the existing datasets
        existing_datasets = json.loads(dataset_config["datasets"])

        # Remove the previously added dataset if it exists
        if self.previous_dataset_signature:
            existing_datasets["datasets"] = [
                ds for ds in existing_datasets["datasets"]
                if self.generate_signature(ds) != self.previous_dataset_signature
            ]

        # Add the new dataset
        existing_datasets["datasets"].append(new_dataset)

        # Store the new dataset signature for future runs
        self.previous_dataset_signature = new_dataset_signature

        # Convert back to JSON and update dataset_config
        updated_dataset_json = json.dumps(existing_datasets, indent=2)
        dataset_config["datasets"] = updated_dataset_json

        return dataset_config,

    def generate_signature(self, dataset):
        # Create a unique signature for the dataset based on its attributes
        return json.dumps(dataset, sort_keys=True)

class OptimizerConfig:
    """这些类主要用于配置不同类型的优化器参数
输入: min_snr_gamma, extra_optimizer_args, **kwargs
- Min SNR gamma 
在 LoRA 学习中，通过在训练图像上添加各种强度的噪声来进行学习（关于此的详细信息省略），但根据放置的噪声强度的差异，学习将通过接近或远离学习目标来保持稳定。因此，引入了 Min SNR gamma 来补偿这一点。特别是当学习带有少量噪声的图像时，可能会与目标偏差很大，因此尝试抑制这种跳跃。
我不会深入细节，因为这很令人困惑，但你可以将此值设置为 0 到 20，默认为 0。
根据论文，最佳值是 5。

内部逻辑: 创建一个包含优化器设置的字典。
输出: 包含优化器设置的字典。
使用时机: 在配置优化器时使用。

    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optimizer_type": (["adamw8bit", "adamw","prodigy", "CAME", "Lion8bit", "Lion", "adamwschedulefree", "sgdschedulefree", "AdEMAMix8bit", "PagedAdEMAMix8bit", "ProdigyPlusScheduleFree"], {"default": "adamw8bit", "tooltip": "optimizer type"}),
            "max_grad_norm": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "gradient clipping"}),
            "lr_scheduler": (["constant", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"], {"default": "constant", "tooltip": "learning rate scheduler"}),
            "lr_warmup_steps": ("INT",{"default": 0, "min": 0, "tooltip": "learning rate warmup steps"}),
            "lr_scheduler_num_cycles": ("INT",{"default": 1, "min": 1, "tooltip": "learning rate scheduler num cycles"}),
            "lr_scheduler_power": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "learning rate scheduler power"}),
            "min_snr_gamma": ("FLOAT",{"default": 5.0, "min": 0.0, "step": 0.01, "tooltip": "gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by the paper"}),
            "extra_optimizer_args": ("STRING",{"multiline": True, "default": "", "tooltip": "additional optimizer args"}),
           },
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("optimizer_settings",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, min_snr_gamma, extra_optimizer_args, **kwargs):
        kwargs["min_snr_gamma"] = min_snr_gamma if min_snr_gamma != 0.0 else None
        kwargs["optimizer_args"] = [arg.strip() for arg in extra_optimizer_args.strip().split('|') if arg.strip()]
        return (kwargs,)

class OptimizerConfigAdafactor:
    """主要功能
配置Adafactor优化器设置。

函数 create_config
输入: relative_step, scale_parameter, warmup_init, clip_threshold, min_snr_gamma, extra_optimizer_args, **kwargs

内部逻辑: 创建一个包含Adafactor优化器设置的字典。

输出: 包含Adafactor优化器设置的字典。

使用时机: 在配置Adafactor优化器时使用。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "max_grad_norm": ("FLOAT",{"default": 0.0, "min": 0.0, "tooltip": "gradient clipping"}),
            "lr_scheduler": (["constant", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "adafactor"], {"default": "constant_with_warmup", "tooltip": "learning rate scheduler"}),
            "lr_warmup_steps": ("INT",{"default": 0, "min": 0, "tooltip": "learning rate warmup steps"}),
            "lr_scheduler_num_cycles": ("INT",{"default": 1, "min": 1, "tooltip": "learning rate scheduler num cycles"}),
            "lr_scheduler_power": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "learning rate scheduler power"}),
            "relative_step": ("BOOLEAN",{"default": False, "tooltip": "relative step"}),
            "scale_parameter": ("BOOLEAN",{"default": False, "tooltip": "scale parameter"}),
            "warmup_init": ("BOOLEAN",{"default": False, "tooltip": "warmup init"}),
            "clip_threshold": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "clip threshold"}),
            "min_snr_gamma": ("FLOAT",{"default": 5.0, "min": 0.0, "step": 0.01, "tooltip": "gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by the paper"}),
            "extra_optimizer_args": ("STRING",{"multiline": True, "default": "", "tooltip": "additional optimizer args"}),
           },
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("optimizer_settings",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, relative_step, scale_parameter, warmup_init, clip_threshold, min_snr_gamma, extra_optimizer_args, **kwargs):
        kwargs["optimizer_type"] = "adafactor"
        extra_args = [arg.strip() for arg in extra_optimizer_args.strip().split('|') if arg.strip()]
        node_args = [
                f"relative_step={relative_step}",
                f"scale_parameter={scale_parameter}",
                f"warmup_init={warmup_init}",
                f"clip_threshold={clip_threshold}"
            ]
        kwargs["optimizer_args"] = node_args + extra_args
        kwargs["min_snr_gamma"] = min_snr_gamma if min_snr_gamma != 0.0 else None
        
        return (kwargs,)

class OptimizerConfigProdigy:
    """主要功能
配置Prodigy优化器设置。

函数 create_config
输入: weight_decay, decouple, min_snr_gamma, use_bias_correction, extra_optimizer_args, **kwargs

内部逻辑: 创建一个包含Prodigy优化器设置的字典。

输出: 包含Prodigy优化器设置的字典。

使用时机: 在配置Prodigy优化器时使用。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "max_grad_norm": ("FLOAT",{"default": 0.0, "min": 0.0, "tooltip": "gradient clipping"}),
            "lr_scheduler": (["constant", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "adafactor"], {"default": "constant", "tooltip": "learning rate scheduler"}),
            "lr_warmup_steps": ("INT",{"default": 0, "min": 0, "tooltip": "learning rate warmup steps"}),
            "lr_scheduler_num_cycles": ("INT",{"default": 1, "min": 1, "tooltip": "learning rate scheduler num cycles"}),
            "lr_scheduler_power": ("FLOAT",{"default": 1.0, "min": 0.0, "tooltip": "learning rate scheduler power"}),
            "weight_decay": ("FLOAT",{"default": 0.0, "step": 0.0001, "tooltip": "weight decay (L2 penalty)"}),
            "decouple": ("BOOLEAN",{"default": True, "tooltip": "use AdamW style weight decay"}),
            "use_bias_correction": ("BOOLEAN",{"default": False, "tooltip": "turn on Adam's bias correction"}),
            "min_snr_gamma": ("FLOAT",{"default": 5.0, "min": 0.0, "step": 0.01, "tooltip": "gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by the paper"}),
            "extra_optimizer_args": ("STRING",{"multiline": True, "default": "", "tooltip": "additional optimizer args"}),
           },
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("optimizer_settings",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, weight_decay, decouple, min_snr_gamma, use_bias_correction, extra_optimizer_args, **kwargs):
        kwargs["optimizer_type"] = "prodigy"
        extra_args = [arg.strip() for arg in extra_optimizer_args.strip().split('|') if arg.strip()]
        node_args = [
                f"weight_decay={weight_decay}",
                f"decouple={decouple}",
                f"use_bias_correction={use_bias_correction}"
            ]
        kwargs["optimizer_args"] = node_args + extra_args
        kwargs["min_snr_gamma"] = min_snr_gamma if min_snr_gamma != 0.0 else None
        
        return (kwargs,)
    
class OptimizerConfigProdigyPlusScheduleFree:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "lr": ("FLOAT",{"default": 1.0, "min": 0.0, "step": 1e-7, "tooltip": "Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate."}),
            "max_grad_norm": ("FLOAT",{"default": 0.0, "min": 0.0, "tooltip": "gradient clipping"}),
            "prodigy_steps": ("INT",{"default": 0, "min": 0, "tooltip": "Freeze Prodigy stepsize adjustments after a certain optimiser step."}),
            "d0": ("FLOAT",{"default": 1e-6, "min": 0.0,"step": 1e-7, "tooltip": "initial learning rate"}),
            "d_coeff": ("FLOAT",{"default": 1.0, "min": 0.0, "step": 1e-7, "tooltip": "Coefficient in the expression for the estimate of d (default 1.0). Values such as 0.5 and 2.0 typically work as well. Changing this parameter is the preferred way to tune the method."}),
            "split_groups": ("BOOLEAN",{"default": True, "tooltip": "Track individual adaptation values for each parameter group."}),
            #"beta3": ("FLOAT",{"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": " Coefficient for computing the Prodigy stepsize using running averages. If set to None, uses the value of square root of beta2 (default: None)."}),
            #"beta4": ("FLOAT",{"default": 0, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Coefficient for updating the learning rate from Prodigy's adaptive stepsize. Smooths out spikes in learning rate adjustments. If set to None, beta1 is used instead. (default 0, which disables smoothing and uses original Prodigy behaviour)."}),
            "use_bias_correction": ("BOOLEAN",{"default": False, "tooltip": "Turn on Adafactor-style bias correction, which scales beta2 directly."}),
            "min_snr_gamma": ("FLOAT",{"default": 5.0, "min": 0.0, "step": 0.01, "tooltip": "gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by the paper"}),
            "extra_optimizer_args": ("STRING",{"multiline": True, "default": "", "tooltip": "additional optimizer args"}),
           
           },
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("optimizer_settings",)
    FUNCTION = "create_config"
    CATEGORY = "FluxTrainer"

    def create_config(self, min_snr_gamma, use_bias_correction, extra_optimizer_args, **kwargs):
        kwargs["optimizer_type"] = "ProdigyPlusScheduleFree"
        kwargs["lr_scheduler"] = "constant"
        extra_args = [arg.strip() for arg in extra_optimizer_args.strip().split('|') if arg.strip()]
        node_args = [
                f"use_bias_correction={use_bias_correction}",
            ]
        kwargs["optimizer_args"] = node_args + extra_args
        kwargs["min_snr_gamma"] = min_snr_gamma if min_snr_gamma != 0.0 else None
        
        return (kwargs,)    

class InitFluxLoRATraining:
    """主要功能
初始化Flux LoRA模型的训练。主要涉及模型选择、数据集配置、优化器设置和输出目录等信息。每个参数都有其特定的作用，用于控制训练过程中的各个方面。
函数内部处理了缓存清理、磁盘空间检查以及各种配置的初始化，并最终返回一个包含网络训练器和训练循环的字典对象。

输入: 
- flux_models
包含训练所需的模型路径的字典，包括transformer、vae、clip_l、t5和可选的lora_path。

- dataset
数据集配置信息，通常包含数据增强设置和其他数据集相关的参数。

- optimizer_settings
优化器设置参数，例如优化器类型、学习率调度器等。

- output_name
输出模型文件的基本名称

- output_dir
模型输出目录路径。默认为 "flux_trainer_output"，根目录是 'ComfyUI' 文件夹，在 Windows 移动版中是 'ComfyUI_windows_portable'。

- network_dim
网络维度，用于定义LoRA网络的维度。

- network_alpha
网络alpha，用于定义LoRA网络的alpha值。

- learning_rate
指定学习率。"学习"是指改变神经网络中线路的厚度（权重），以便制作出与给定图片完全相同的图片，但每次给定一张图片，线路都会改变。如果你只调整给定的图片，将无法绘制其他任何图片。

- max_train_steps
说明: 最大训练步数，定义训练过程中的最大步数。

- apply_t5_attn_mask
是否应用t5注意力掩码。

- cache_latents
缓存潜变量的方式，可以选择磁盘、内存或禁用。

- cache_text_encoder_outputs
缓存文本编码器输出的方式，可以选择磁盘、内存或禁用。

- split_mode
是否使用分割模式，实验性功能。

- weighting_scheme
权重方案，定义如何计算权重

- logit_mean
logit_normal权重方案的均值。

- logit_std
logit_normal权重方案的标准差。

- mode_scale
模式权重方案的缩放

- timestep_sampling
时间步采样方法，定义如何采样时间步。

- sigmoid_scale
sigmoid时间步采样的缩放因子。

- model_prediction_type
模型预测类型，定义如何解释和处理模型预测。

- guidance_scale
指导缩放，用于Flux训练。

- discrete_flow_shift
Euler离散调度器的偏移。

- highvram
是否使用高VRAM模式。

- fp8_base
是否使用fp8作为基础模型。

- gradient_dtype
梯度数据类型，定义训练过程中使用的数据类型。

- save_dtype
保存检查点的数据类型，定义保存检查点时使用的数据类型。

- attention_mode
注意力模式，定义如何计算注意力。

- sample_prompts
验证样本提示，用于验证过程中的提示。

- additional_args
额外的训练命令参数，用于传递额外的训练参数。

- resume_args
恢复训练的参数，用于恢复训练过程。

- train_text_encoder
是否训练文本编码器，定义训练过程中是否训练文本编码器。

- clip_l_lr
clip_l文本编码器的学习率。

- T5_lr
T5文本编码器的学习率。

- block_args
限制LoRA中使用的块，定义哪些块将被用于LoRA训练。

- gradient_checkpointing
是否使用梯度检查点，定义是否启用梯度检查点以及是否使用CPU卸载。

- prompt
提示信息，用于传递额外的提示信息。

- extra_pnginfo
额外的PNG信息，用于传递额外的PNG信息。



内部逻辑: 初始化训练环境，配置模型和优化器，并开始训练。

输出: 训练器对象、训练周期数、训练参数。

使用时机: 在开始LoRA模型训练时使用。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "dataset": ("JSON",),
            "optimizer_settings": ("ARGS",),
            "output_name": ("STRING", {"default": "flux_lora", "multiline": False}),
            "output_dir": ("STRING", {"default": "flux_trainer_output", "multiline": False, "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "network_dim": ("INT", {"default": 4, "min": 1, "max": 2048, "step": 1, "tooltip": "network dim"}),
            "network_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2048.0, "step": 0.01, "tooltip": "network alpha"}),
            "learning_rate": ("FLOAT", {"default": 4e-4, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "learning rate"}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 1, "tooltip": "max number of training steps"}),
            "apply_t5_attn_mask": ("BOOLEAN", {"default": True, "tooltip": "apply t5 attention mask"}),
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "split_mode": ("BOOLEAN", {"default": False, "tooltip": "[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"}),
            "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"],),
            "logit_mean": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean to use when using the logit_normal weighting scheme"}),
            "logit_std": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "std to use when using the logit_normal weighting scheme"}),
            "mode_scale": ("FLOAT", {"default": 1.29, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scale of mode weighting scheme. Only effective when using the mode as the weighting_scheme"}),
            "timestep_sampling": (["sigmoid", "uniform", "sigma", "shift", "flux_shift"], {"tooltip": "Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal and shift of sigmoid (recommend value of 3.1582 for discrete_flow_shift)"}),
            "sigmoid_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Scale factor for sigmoid timestep sampling (only used when timestep-sampling is sigmoid"}),
            "model_prediction_type": (["raw", "additive", "sigma_scaled"], {"tooltip": "How to interpret and process the model prediction: raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."}),
            "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0, "step": 0.01, "tooltip": "guidance scale, for Flux training should be 1.0"}),
            "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "for the Euler Discrete Scheduler, default is 3.0"}),
            "highvram": ("BOOLEAN", {"default": False, "tooltip": "memory mode"}),
            "fp8_base": ("BOOLEAN", {"default": True, "tooltip": "use fp8 for base model"}),
            "gradient_dtype": (["fp32", "fp16", "bf16"], {"default": "fp32", "tooltip": "the actual dtype training uses"}),
            "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "bf16", "tooltip": "the dtype to save checkpoints as"}),
            "attention_mode": (["sdpa", "xformers", "disabled"], {"default": "sdpa", "tooltip": "memory efficient attention mode"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            },
            "optional": {
                "additional_args": ("STRING", {"multiline": True, "default": "", "tooltip": "additional args to pass to the training command"}),
                "resume_args": ("ARGS", {"default": "", "tooltip": "resume args to pass to the training command"}),
                "train_text_encoder": (['disabled', 'clip_l', 'clip_l_fp8', 'clip_l+T5', 'clip_l+T5_fp8'], {"default": 'disabled', "tooltip": "also train the selected text encoders using specified dtype, T5 can not be trained without clip_l"}),
                "clip_l_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "text encoder learning rate"}),
                "T5_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "text encoder learning rate"}),
                "block_args": ("ARGS", {"default": "", "tooltip": "limit the blocks used in the LoRA"}),
                "gradient_checkpointing": (["enabled", "enabled_with_cpu_offloading", "disabled"], {"default": "enabled", "tooltip": "use gradient checkpointing"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "KOHYA_ARGS",)
    RETURN_NAMES = ("network_trainer", "epochs_count", "args",)
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer"

    def init_training(self, flux_models, dataset, optimizer_settings, sample_prompts, output_name, attention_mode, 
                      gradient_dtype, save_dtype, split_mode, additional_args=None, resume_args=None, train_text_encoder='disabled', 
                      block_args=None, gradient_checkpointing="enabled", prompt=None, extra_pnginfo=None, clip_l_lr=0, T5_lr=0, **kwargs,):
        """* 输入
flux_models: Flux模型配置，包含模型的路径和其他相关信息。

dataset: 数据集配置，包含数据集的路径、类标记、重复次数等信息。

optimizer_settings: 优化器设置，包含优化器类型、学习率调度器、梯度裁剪等信息。

output_name: 输出模型的名称。

output_dir: 输出目录的路径。

network_dim: 网络维度，用于定义LoRA网络的维度。

network_alpha: 网络alpha，用于定义LoRA网络的alpha值。

learning_rate: 学习率，用于定义训练过程中的学习率。

max_train_steps: 最大训练步数，定义训练过程中的最大步数。

apply_t5_attn_mask: 是否应用t5注意力掩码。

cache_latents: 缓存潜变量的方式，可以选择磁盘、内存或禁用。

cache_text_encoder_outputs: 缓存文本编码器输出的方式，可以选择磁盘、内存或禁用。

split_mode: 是否使用分割模式，实验性功能。

weighting_scheme: 权重方案，定义如何计算权重。

logit_mean: logit_normal权重方案的均值。

logit_std: logit_normal权重方案的标准差。

mode_scale: 模式权重方案的缩放。

timestep_sampling: 时间步采样方法，定义如何采样时间步。

sigmoid_scale: sigmoid时间步采样的缩放因子。

model_prediction_type: 模型预测类型，定义如何解释和处理模型预测。

guidance_scale: 指导缩放，用于Flux训练。

discrete_flow_shift: Euler离散调度器的偏移。

highvram: 是否使用高VRAM模式。

fp8_base: 是否使用fp8作为基础模型。

gradient_dtype: 梯度数据类型，定义训练过程中使用的数据类型。

save_dtype: 保存检查点的数据类型，定义保存检查点时使用的数据类型。

attention_mode: 注意力模式，定义如何计算注意力。

sample_prompts: 验证样本提示，用于验证过程中的提示。

additional_args: 额外的训练命令参数，用于传递额外的训练参数。

resume_args: 恢复训练的参数，用于恢复训练过程。

train_text_encoder: 是否训练文本编码器，定义训练过程中是否训练文本编码器。

clip_l_lr: clip_l文本编码器的学习率。

T5_lr: T5文本编码器的学习率。

block_args: 限制LoRA中使用的块，定义哪些块将被用于LoRA训练。

gradient_checkpointing: 是否使用梯度检查点，定义是否启用梯度检查点以及是否使用CPU卸载。

prompt: 提示信息，用于传递额外的提示信息。

extra_pnginfo: 额外的PNG信息，用于传递额外的PNG信息。

* 内部逻辑
初始化训练环境:

创建输出目录。

检查磁盘空间是否足够。

加载数据集配置:

将数据集配置转换为TOML格式。

解析命令行参数:

解析额外的训练命令参数。

配置缓存选项:

根据缓存选项配置缓存潜变量和文本编码器输出。

处理验证样本提示:

将验证样本提示分割为多个提示。

配置模型和优化器:

创建包含模型和优化器配置的字典。

根据注意力模式、梯度数据类型和保存数据类型更新配置。

如果训练文本编码器，设置文本编码器的学习率。

处理网络参数:

处理额外的网络参数。

保存训练参数:

将训练参数保存为JSON文件。

保存工作流:

将工作流保存为JSON文件。

初始化训练器:

初始化FluxNetworkTrainer对象。

开始训练。

* 输出
训练器对象: 包含训练器和训练循环的对象。

训练周期数: 训练的总周期数。

训练参数: 包含所有训练参数的对象。"""

        """1. 清理缓存，释放内存。
        缓存清理: 使用 mm.soft_empty_cache() 清理缓存，确保有足够的内存用于训练。
        创建输出目录: 检查并创建输出目录，确保路径存在。
        磁盘空间检查: 检查输出目录的可用空间是否足够，如果不足则抛出异常。
                """
        mm.soft_empty_cache()
        

        output_dir = os.path.abspath(kwargs.get("output_dir"))
        os.makedirs(output_dir, exist_ok=True)
    
        total, used, free = shutil.disk_usage(output_dir)
 
        required_free_space = 2 * (2**30)
        if free <= required_free_space:
            raise ValueError(f"Insufficient disk space. Required: {required_free_space/2**30}GB. Available: {free/2**30}GB")
        """2. 数据集配置转换
        数据集配置转换: 将数据集配置从 JSON 格式转换为 TOML 格式，以便后续使用。"""
        dataset_config = dataset["datasets"]
        dataset_toml = toml.dumps(json.loads(dataset_config))

        """3. 解析命令行参数
        解析命令行参数: 使用 train_network_setup_parser() 创建解析器，并解析额外的训练命令参数。"""
        parser = train_network_setup_parser()
        if additional_args is not None:
            print(f"additional_args: {additional_args}")
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
        else:
            args, _ = parser.parse_known_args()

        """4. 配置缓存选项
        根据 cache_latents 和 cache_text_encoder_outputs 的值设置相应的缓存选项。
        如果缓存到磁盘，则禁用某些数据增强选项。"""
        if kwargs.get("cache_latents") == "memory":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = False
        elif kwargs.get("cache_latents") == "disk":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = True
            # 为什么将隐空间存储在磁盘里面会将标签丢弃率设置为0
            kwargs["caption_dropout_rate"] = 0.0
            kwargs["shuffle_caption"] = False
            kwargs["token_warmup_step"] = 0.0
            kwargs["caption_tag_dropout_rate"] = 0.0
        else:
            kwargs["cache_latents"] = False
            kwargs["cache_latents_to_disk"] = False

        if kwargs.get("cache_text_encoder_outputs") == "memory":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = False
        elif kwargs.get("cache_text_encoder_outputs") == "disk":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = True
        else:
            kwargs["cache_text_encoder_outputs"] = False
            kwargs["cache_text_encoder_outputs_to_disk"] = False

        """5. 处理验证样本提示
        如果 sample_prompts 中包含 |，则将其分割为多个提示。
        否则，将其作为一个提示列表。
"""
        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        """6. 配置模型和优化器
            目的: 创建包含模型和优化器配置的字典。
    运行逻辑:
        初始化 config_dict 字典，包含各种配置参数。
        根据 attention_mode 和 gradient_dtype 更新配置。
        如果训练文本编码器，设置文本编码器的学习率。
        处理额外的网络参数。
        根据 gradient_checkpointing 设置梯度检查点选项。
        如果提供了 lora_path，则将其添加到配置中。
        更新配置字典中的其他参数。
        如果提供了 resume_args，则将其更新到配置中。
        将配置字典中的参数设置到 args 命名空间中。
    """
        config_dict = {
            "sample_prompts": prompts,
            "save_precision": save_dtype,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": flux_models["transformer"],
            "clip_l": flux_models["clip_l"],
            "t5xxl": flux_models["t5"],
            "ae": flux_models["vae"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "network_module": ".networks.lora_flux",
            "dataset_config": dataset_toml,
            "output_name": f"{output_name}_rank{kwargs.get('network_dim')}_{save_dtype}",
            "loss_type": "l2",
            "t5xxl_max_token_length": 512,
            "alpha_mask": dataset["alpha_mask"],
            "network_train_unet_only": True if train_text_encoder == 'disabled' else False,
            "fp8_base_unet": True if "fp8" in train_text_encoder else False,
            "disable_mmap_load_safetensors": False,
            "split_mode": split_mode,
        }
        attention_settings = {
            "sdpa": {"mem_eff_attn": True, "xformers": False, "spda": True},
            "xformers": {"mem_eff_attn": True, "xformers": True, "spda": False}
        }
        config_dict.update(attention_settings.get(attention_mode, {}))

        gradient_dtype_settings = {
            "fp16": {"full_fp16": True, "full_bf16": False, "mixed_precision": "fp16"},
            "bf16": {"full_bf16": True, "full_fp16": False, "mixed_precision": "bf16"}
        }
        config_dict.update(gradient_dtype_settings.get(gradient_dtype, {}))

        if train_text_encoder != 'disabled':
            if T5_lr != "NaN":
                config_dict["text_encoder_lr"] = clip_l_lr
            if T5_lr != "NaN":
                config_dict["text_encoder_lr"] = [clip_l_lr, T5_lr]

        #network args
        additional_network_args = []
        
        if "T5" in train_text_encoder:
            additional_network_args.append("train_t5xxl=True")
        if split_mode:
            additional_network_args.append("train_blocks=single")
        if block_args:
            additional_network_args.append(block_args["include"])
        ### 添加额外的参数
        # Handle network_args in args Namespace
        if hasattr(args, 'network_args') and isinstance(args.network_args, list):
            args.network_args.extend(additional_network_args)
        else:
            setattr(args, 'network_args', additional_network_args)

        if gradient_checkpointing == "disabled":
            config_dict["gradient_checkpointing"] = False
        elif gradient_checkpointing == "enabled_with_cpu_offloading":
            config_dict["gradient_checkpointing"] = True
            config_dict["cpu_offload_checkpointing"] = True
        else:
            config_dict["gradient_checkpointing"] = True

        if flux_models["lora_path"]:
            config_dict["network_weights"] = flux_models["lora_path"]

        config_dict.update(kwargs)
        config_dict.update(optimizer_settings)

        if resume_args:
            config_dict.update(resume_args)

        for key, value in config_dict.items():
            setattr(args, key, value)
        """7. 创建输出文件夹及保存训练参数文件
        创建两个 JSON 文件来存储当前会话的所有训练信息：一个用于存储所有必要的命令线 参数 (args.json)；另一个用于存储任何附加的工作流元数据(workflow.json)。这些文件对于重现实验非常有用，并且可以方便地查看当前会话的所有关键设置信息。"""
        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        #workflow saving
        metadata = {}
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo["workflow"])
       
        saved_workflow_file_path = os.path.join(output_dir, f"{output_name}_workflow.json")
        with open(saved_workflow_file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        """11. 初始化训练器
        始化FluxNetworkTrainer对象并开始训练。禁用推理模式。
        创建 FluxNetworkTrainer 对象。
        调用 init_train 方法初始化训练器并开始训练。"""
        #pass args to kohya and initialize trainer
        with torch.inference_mode(False):
            network_trainer = FluxNetworkTrainer()
            training_loop = network_trainer.init_train(args)

        """12. 开始训练"""
        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)

class InitFluxTraining:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "dataset": ("JSON",),
            "optimizer_settings": ("ARGS",),
            "output_name": ("STRING", {"default": "flux", "multiline": False}),
            "output_dir": ("STRING", {"default": "flux_trainer_output", "multiline": False, "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "learning_rate": ("FLOAT", {"default": 4e-6, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "learning rate"}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 1, "tooltip": "max number of training steps"}),
            "apply_t5_attn_mask": ("BOOLEAN", {"default": True, "tooltip": "apply t5 attention mask"}),
            "t5xxl_max_token_length": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "dev and LibreFlux uses 512, schnell 256"}),
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "weighting_scheme": (["logit_normal", "sigma_sqrt", "mode", "cosmap", "none"],),
            "logit_mean": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean to use when using the logit_normal weighting scheme"}),
            "logit_std": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "std to use when using the logit_normal weighting scheme"}),
            "mode_scale": ("FLOAT", {"default": 1.29, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scale of mode weighting scheme. Only effective when using the mode as the weighting_scheme"}),
            "loss_type": (["l1", "l2", "huber", "smooth_l1"], {"default": "l2", "tooltip": "loss type"}),
            "timestep_sampling": (["sigmoid", "uniform", "sigma", "shift", "flux_shift"], {"tooltip": "Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal and shift of sigmoid (recommend value of 3.1582 for discrete_flow_shift)"}),
            "sigmoid_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Scale factor for sigmoid timestep sampling (only used when timestep-sampling is sigmoid"}),
            "model_prediction_type": (["raw", "additive", "sigma_scaled"], {"tooltip": "How to interpret and process the model prediction: raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)"}),
            "cpu_offload_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "offload the gradient checkpointing to CPU. This reduces VRAM usage for about 2GB"}),
            "optimizer_fusing": (['fused_backward_pass', 'blockwise_fused_optimizers'], {"tooltip": "reduces memory use"}),
            "single_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "number of single blocks to swap. The default is 0. This option must be combined with blockwise_fused_optimizers"}),
            "double_blocks_to_swap": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1, "tooltip": "number of double blocks to swap. This option must be combined with blockwise_fused_optimizers"}),
            "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0, "step": 0.01, "tooltip": "guidance scale"}),
            "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "for the Euler Discrete Scheduler, default is 3.0"}),
            "highvram": ("BOOLEAN", {"default": False, "tooltip": "memory mode"}),
            "fp8_base": ("BOOLEAN", {"default": False, "tooltip": "use fp8 for base model"}),
            "gradient_dtype": (["fp32", "fp16", "bf16"], {"default": "bf16", "tooltip": "to use the full fp16/bf16 training"}),
            "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn"], {"default": "bf16", "tooltip": "the dtype to save checkpoints as"}),
            "attention_mode": (["sdpa", "xformers", "disabled"], {"default": "sdpa", "tooltip": "memory efficient attention mode"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            },
            "optional": {
                "additional_args": ("STRING", {"multiline": True, "default": "", "tooltip": "additional args to pass to the training command"}),
                "resume_args": ("ARGS", {"default": "", "tooltip": "resume args to pass to the training command"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "KOHYA_ARGS")
    RETURN_NAMES = ("network_trainer", "epochs_count", "args")
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer"

    def init_training(self, flux_models, optimizer_settings, dataset, sample_prompts, output_name, 
                      attention_mode, gradient_dtype, save_dtype, optimizer_fusing, additional_args=None, resume_args=None, **kwargs,):
        """flux_models: Flux模型配置，包含模型的路径和其他相关信息。

optimizer_settings: 优化器设置，包含优化器类型、学习率调度器、梯度裁剪等信息。

dataset: 数据集配置，包含数据集的路径、类标记、重复次数等信息。

sample_prompts: 验证样本提示，用于验证过程中的提示。

output_name: 输出模型的名称。

attention_mode: 注意力模式，定义如何计算注意力。

gradient_dtype: 梯度数据类型，定义训练过程中使用的数据类型。

save_dtype: 保存检查点的数据类型，定义保存检查点时使用的数据类型。

optimizer_fusing: 优化器融合方式"""
        mm.soft_empty_cache()

        output_dir = os.path.abspath(kwargs.get("output_dir"))
        os.makedirs(output_dir, exist_ok=True)
    
        total, used, free = shutil.disk_usage(output_dir)
        required_free_space = 25 * (2**30)
        if free <= required_free_space:
            raise ValueError(f"Most likely insufficient disk space to complete training. Required: {required_free_space/2**30}GB. Available: {free/2**30}GB")

        dataset_config = dataset["datasets"]
        dataset_toml = toml.dumps(json.loads(dataset_config))
        
        parser = train_setup_parser()
        if additional_args is not None:
            print(f"additional_args: {additional_args}")
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
        else:
            args, _ = parser.parse_known_args()

        if kwargs.get("cache_latents") == "memory":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = False
        elif kwargs.get("cache_latents") == "disk":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = True
            kwargs["caption_dropout_rate"] = 0.0
            kwargs["shuffle_caption"] = False
            kwargs["token_warmup_step"] = 0.0
            kwargs["caption_tag_dropout_rate"] = 0.0
        else:
            kwargs["cache_latents"] = False
            kwargs["cache_latents_to_disk"] = False

        if kwargs.get("cache_text_encoder_outputs") == "memory":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = False
        elif kwargs.get("cache_text_encoder_outputs") == "disk":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = True
        else:
            kwargs["cache_text_encoder_outputs"] = False
            kwargs["cache_text_encoder_outputs_to_disk"] = False

        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        config_dict = {
            "sample_prompts": prompts,
            "save_precision": save_dtype,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": flux_models["transformer"],
            "clip_l": flux_models["clip_l"],
            "t5xxl": flux_models["t5"],
            "ae": flux_models["vae"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "gradient_checkpointing": True,
            "dataset_config": dataset_toml,
            "output_name": f"{output_name}_{save_dtype}",
            "mem_eff_save": True,
            "disable_mmap_load_safetensors": True,

        }
        optimizer_fusing_settings = {
            "fused_backward_pass": {"fused_backward_pass": True, "blockwise_fused_optimizers": False},
            "blockwise_fused_optimizers": {"fused_backward_pass": False, "blockwise_fused_optimizers": True}
        }
        config_dict.update(optimizer_fusing_settings.get(optimizer_fusing, {}))

        attention_settings = {
            "sdpa": {"mem_eff_attn": True, "xformers": False, "spda": True},
            "xformers": {"mem_eff_attn": True, "xformers": True, "spda": False}
        }
        config_dict.update(attention_settings.get(attention_mode, {}))

        gradient_dtype_settings = {
            "fp16": {"full_fp16": True, "full_bf16": False, "mixed_precision": "fp16"},
            "bf16": {"full_bf16": True, "full_fp16": False, "mixed_precision": "bf16"}
        }
        config_dict.update(gradient_dtype_settings.get(gradient_dtype, {}))

        config_dict.update(kwargs)
        config_dict.update(optimizer_settings)

        if resume_args:
            config_dict.update(resume_args)

        for key, value in config_dict.items():
            setattr(args, key, value)

        with torch.inference_mode(False):
            network_trainer = FluxTrainer()
            training_loop = network_trainer.init_train(args)

        epochs_count = network_trainer.num_train_epochs

        
        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)

class InitFluxTrainingFromPreset:
    """
    **参数说明:**

1. **flux_models**:
   - **类型**: `TRAIN_FLUX_MODELS`
   - **描述**: 包含训练所需的模型路径的字典，包括transformer、vae、clip_l、t5和可选的lora_path。
   - **输入**: 必需

2. **dataset_settings**:
   - **类型**: `TOML_DATASET`
   - **描述**: 数据集设置信息，通常以 TOML 格式存储。
   - **输入**: 必需

3. **preset_args**:
   - **类型**: `KOHYA_ARGS`
   - **描述**: 预设的训练参数，通常包含优化器设置和其他训练配置。
   - **输入**: 必需

4. **output_name**:
   - **类型**: `STRING`
   - **默认值**: `"flux"`
   - **描述**: 输出模型文件的基本名称。

5. **output_dir**:
   - **类型**: `STRING`
   - **默认值**: `"flux_trainer_output"`
   - 描述: 模型输出目录路径。默认为 `"flux_trainer_output"`，根目录是 `'ComfyUI'` 文件夹。

6. **sample_prompts**:
   - 类型: `STRING`
   - 默认值: `"illustration of a kitten | photograph of a turtle"`
   - 描述: 验证样本提示，多个提示用 `|` 分隔。

* 内部运行逻辑

初始化训练环境:
创建输出目录。
检查磁盘空间是否足够。
加载数据集配置:
将数据集配置转换为TOML格式。
解析命令行参数:
解析预设参数。
处理验证样本提示:
将验证样本提示分割为多个提示。
配置模型和优化器:
创建包含模型和优化器配置的字典。
根据注意力模式、梯度数据类型和保存数据类型更新配置。
保存训练参数:
将训练参数保存为JSON文件。
初始化训练器:
初始化FluxNetworkTrainer对象。
开始训练。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "dataset_settings": ("TOML_DATASET",),
            "preset_args": ("KOHYA_ARGS",),
            "output_name": ("STRING", {"default": "flux", "multiline": False}),
            "output_dir": ("STRING", {"default": "flux_trainer_output", "multiline": False, "tooltip": "output directory, root is ComfyUI folder"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "STRING", "KOHYA_ARGS")
    RETURN_NAMES = ("network_trainer", "epochs_count", "output_path", "args")
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer"

    def init_training(self, flux_models, dataset_settings, sample_prompts, output_name, preset_args, **kwargs,):
        mm.soft_empty_cache()

        dataset = dataset_settings["dataset"]
        dataset_repeats = dataset_settings["repeats"]
        
        parser = train_setup_parser()
        args, _ = parser.parse_known_args()
        for key, value in vars(preset_args).items():
            setattr(args, key, value)
        
        output_dir = os.path.join(script_directory, "output")
        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        width, height = toml.loads(dataset)["datasets"][0]["resolution"]
        config_dict = {
            "sample_prompts": prompts,
            "dataset_repeats": dataset_repeats,
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": flux_models["transformer"],
            "clip_l": flux_models["clip_l"],
            "t5xxl": flux_models["t5"],
            "ae": flux_models["vae"],
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "gradient_checkpointing": True,
            "dataset_config": dataset,
            "output_dir": output_dir,
            "output_name": f"{output_name}_rank{kwargs.get('network_dim')}_{args.save_precision}",
            "width" : int(width),
            "height" : int(height),

        }

        config_dict.update(kwargs)

        for key, value in config_dict.items():
            setattr(args, key, value)

        with torch.inference_mode(False):
            network_trainer = FluxNetworkTrainer()
            training_loop = network_trainer.init_train(args)

        final_output_path = os.path.join(output_dir, output_name)

        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, final_output_path, args)
    
class FluxTrainLoop:
    """**参数说明:**

1. **network_trainer**:
   - **类型**: `NETWORKTRAINER`
   - **描述**: 包含网络训练器和训练循环的字典对象。
   - **输入**: 必需

2. **steps**:
   - **类型**: `INT`
   - **默认值**: `1`
   - 范围: `[1, 10000]`
   - 步长: `1`
   - 描述: 训练步数，用于指定在训练过程中进行验证或保存的步数点。

* 内部运行逻辑
1. 初始化训练环境:
获取训练器和训练循环对象。

2. 设置目标全局步数:
计算目标全局步数，即当前全局步数加上训练步数。

3. 创建进度条:
创建一个进度条，用于显示训练进度。

4. 执行训练循环:
执行指定步数的训练循环。
5. 更新进度条。
检查全局步数是否达到最大训练步数。

6. 返回训练器和当前全局步数:
返回更新后的训练器对象和当前全局步数。
"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "steps": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1, "tooltip": "the step point in training to validate/save"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT",)
    RETURN_NAMES = ("network_trainer", "steps",)
    FUNCTION = "train"
    CATEGORY = "FluxTrainer"

    def train(self, network_trainer, steps):
        """该函数的主要目的是执行训练循环，并在指定的步数内进行训练。它通过更新训练器对象和当前全局步数来实现这一目标。"""
        """1. 初始化训练环境
        确保处于训练模式（非推理模式）。
        获取训练器和训练循环对象。
        获取当前全局步数。
        """
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]
            initial_global_step = network_trainer.global_step

            """2. 设置目标全局步数
            计算目标全局步数，即当前全局步数加上训练步数。确定在本次调用中需要达到的目标全局步数。"""
            target_global_step = network_trainer.global_step + steps

            """3. 创建进度条
            创建一个进度条，用于显示训练进度。"""
            comfy_pbar = comfy.utils.ProgressBar(steps)
            network_trainer.comfy_pbar = comfy_pbar

            """4. 执行优化器函数并开始循环
            调用 network_trainer.optimizer_train_fn() 初始化优化器。
            进入循环，直到 network_trainer.global_step 达到 target_global_step。
            在每次循环中，调用 training_loop 函数进行训练，并传入 break_at_steps 和 epoch 参数。
            如果 network_trainer.global_step 达到 network_trainer.args.max_train_steps，则跳出循环。"""
            network_trainer.optimizer_train_fn()

            while network_trainer.global_step < target_global_step:
                steps_done = training_loop(
                    break_at_steps = target_global_step,
                    epoch = network_trainer.current_epoch.value,
                )
                #pbar.update(steps_done)
               
                # Also break if the global steps have reached the max train steps
                if network_trainer.global_step >= network_trainer.args.max_train_steps:
                    break
            
            trainer = {
                "network_trainer": network_trainer,
                "training_loop": training_loop,
            }
        return (trainer, network_trainer.global_step)

class FluxTrainAndValidateLoop:
    """**参数说明:**

1. **network_trainer**:
   - **类型**: `NETWORKTRAINER`
   - **描述**: 包含网络训练器和训练循环的字典对象。
   - **输入**: 必需

2. **validate_at_steps**:
   - **类型**: `INT`
   - **默认值**: `250`
   - 范围: `[1, 10000]`
   - 步长: `1`
   - 描述: 在训练过程中进行验证的步数点。

3. **save_at_steps**:
   - **类型**: `INT`
   - **默认值**: `250`
   - 范围: `[1, 10000]`
   - 步长: `1`
   - 描述: 在训练过程中进行保存的步数点。

4. **validation_settings**:
   - **类型**: `VALSETTINGS`（可选）
   - 描述: 验证设置信息，包含验证相关的配置参数。   

* 内部运行逻辑
1. 初始化训练环境:
获取训练器和训练循环对象。
2. 设置目标全局步数:
计算目标全局步数，即最大训练步数。
3. 创建进度条:
创建一个进度条，用于显示训练进度。
4. 执行训练循环:
执行训练循环，直到达到最大训练步数。
计算下一个验证步数和保存步数。
更新进度条。
检查全局步数是否达到验证步数或保存步数。
5. 验证和保存模型:
如果达到验证步数，执行验证。
如果达到保存步数，保存模型。
6. 返回训练器和当前全局步数:
返回更新后的训练器对象和当前全局步数。
   """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "validate_at_steps": ("INT", {"default": 250, "min": 1, "max": 10000, "step": 1, "tooltip": "the step point in training to validate/save"}),
            "save_at_steps": ("INT", {"default": 250, "min": 1, "max": 10000, "step": 1, "tooltip": "the step point in training to validate/save"}),
            },
             "optional": {
                "validation_settings": ("VALSETTINGS",),
            }
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT",)
    RETURN_NAMES = ("network_trainer", "steps",)
    FUNCTION = "train"
    CATEGORY = "FluxTrainer"

    def train(self, network_trainer, validate_at_steps, save_at_steps, validation_settings=None):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]

            target_global_step = network_trainer.args.max_train_steps
            comfy_pbar = comfy.utils.ProgressBar(target_global_step)
            network_trainer.comfy_pbar = comfy_pbar

            network_trainer.optimizer_train_fn()

            while network_trainer.global_step < target_global_step:
                next_validate_step = ((network_trainer.global_step // validate_at_steps) + 1) * validate_at_steps
                next_save_step = ((network_trainer.global_step // save_at_steps) + 1) * save_at_steps

                steps_done = training_loop(
                    break_at_steps=min(next_validate_step, next_save_step),
                    epoch=network_trainer.current_epoch.value,
                )

                # Check if we need to validate
                if network_trainer.global_step % validate_at_steps == 0:
                    self.validate(network_trainer, validation_settings)

                # Check if we need to save
                if network_trainer.global_step % save_at_steps == 0:
                    self.save(network_trainer)

                # Also break if the global steps have reached the max train steps
                if network_trainer.global_step >= network_trainer.args.max_train_steps:
                    break

            trainer = {
                "network_trainer": network_trainer,
                "training_loop": training_loop,
            }
        return (trainer, network_trainer.global_step)

    def validate(self, network_trainer, validation_settings=None):
        params = (
            network_trainer.accelerator, 
            network_trainer.args, 
            network_trainer.current_epoch.value, 
            network_trainer.global_step,
            network_trainer.unet,
            network_trainer.vae,
            network_trainer.text_encoder,
            network_trainer.sample_prompts_te_outputs,
            validation_settings
        )
        network_trainer.optimizer_eval_fn()
        image_tensors = network_trainer.sample_images(*params)
        network_trainer.optimizer_train_fn()
        print("Validating at step:", network_trainer.global_step)

    def save(self, network_trainer):
        ckpt_name = train_util.get_step_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as, network_trainer.global_step)
        network_trainer.optimizer_eval_fn()
        network_trainer.save_model(ckpt_name, network_trainer.accelerator.unwrap_model(network_trainer.network), network_trainer.global_step, network_trainer.current_epoch.value + 1)
        network_trainer.optimizer_train_fn()
        print("Saving at step:", network_trainer.global_step)

class FluxTrainSave:
    """FluxTrainSave 类提供了一个全面的框架，用于保存训练模型。通过配置训练器、是否保存整个模型状态以及是否将LoRA模型复制到comfy lora文件夹
1. **network_trainer**:
   - **类型**: `NETWORKTRAINER`
   - **描述**: 包含网络训练器和训练循环的字典对象。
   - **输入**: 必需

2. **save_at_steps**:
   - **类型**: `INT`
   - **默认值**: `250`
   - 范围: `[1, 10000]`
   - 步长: `1`
   - 描述: 在训练过程中进行保存的步数点。
   
* 内部运行逻辑
1. 初始化训练环境:
获取训练器对象。

2. 获取当前全局步数:
获取当前训练过程中的全局步数。

3. 保存模型:
根据当前全局步数生成检查点名称。
保存模型权重。

4. 删除旧的检查点:
计算需要删除的旧检查点步数。
删除旧的检查点文件。

5. 保存模型状态:
如果 save_state 为 True，保存整个模型状态。

6. 复制LoRA模型到comfy lora文件夹:
如果 copy_to_comfy_lora_folder 为 True，将LoRA模型复制到comfy lora文件夹。

7. 返回训练器、LoRA路径和当前全局步数:
返回更新后的训练器对象、LoRA路径和当前全局步数。
   """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": False, "tooltip": "save the whole model state as well"}),
            "copy_to_comfy_lora_folder": ("BOOLEAN", {"default": False, "tooltip": "copy the lora model to the comfy lora folder"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING", "INT",)
    RETURN_NAMES = ("network_trainer","lora_path", "steps",)
    FUNCTION = "save"
    CATEGORY = "FluxTrainer"

    def save(self, network_trainer, save_state, copy_to_comfy_lora_folder):
        import shutil
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step
            
            ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, global_step)
            trainer.save_model(ckpt_name, trainer.accelerator.unwrap_model(trainer.network), global_step, trainer.current_epoch.value + 1)

            remove_step_no = train_util.get_remove_step_no(trainer.args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, remove_step_no)
                trainer.remove_model(remove_ckpt_name)

            if save_state:
                train_util.save_and_remove_state_stepwise(trainer.args, trainer.accelerator, global_step)

            lora_path = os.path.join(trainer.args.output_dir, ckpt_name)
            if copy_to_comfy_lora_folder:
                destination_dir = os.path.join(folder_paths.models_dir, "loras", "flux_trainer")
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy(lora_path, os.path.join(destination_dir, ckpt_name))
        
            
        return (network_trainer, lora_path, global_step)

class FluxTrainSaveModel:
    """* 参数详细说明
1. network_trainer:

类型: NETWORKTRAINER

说明: 网络训练器对象，包含训练器和训练循环的对象。

2. copy_to_comfy_model_folder:

类型: BOOLEAN

默认值: False

说明: 是否将LoRA模型复制到comfy lora文件夹，以便在其他地方使用。

3. end_training:

类型: BOOLEAN

默认值: False

说明: 是否结束训练，如果为 True，则在保存模型后结束训练。

* 内部逻辑
1. 初始化训练环境:
获取训练器对象。

2. 获取当前全局步数:
获取当前训练过程中的全局步数。

3. 保存模型:
根据当前全局步数生成检查点名称。
保存模型权重。

4. 复制LoRA模型到comfy lora文件夹:
如果 copy_to_comfy_model_folder 为 True，将LoRA模型复制到comfy lora文件夹。

5. 结束训练:
如果 end_training 为 True，结束训练。

6. 返回训练器、模型路径和当前全局步数:
返回更新后的训练器对象、模型路径和当前全局步数。
"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "copy_to_comfy_model_folder": ("BOOLEAN", {"default": False, "tooltip": "copy the lora model to the comfy lora folder"}),
            "end_training": ("BOOLEAN", {"default": False, "tooltip": "end the training"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING", "INT",)
    RETURN_NAMES = ("network_trainer","model_path", "steps",)
    FUNCTION = "save"
    CATEGORY = "FluxTrainer"

    def save(self, network_trainer, copy_to_comfy_model_folder, end_training):
        import shutil
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step

            trainer.optimizer_eval_fn()
            
            ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, global_step)
            flux_train_utils.save_flux_model_on_epoch_end_or_stepwise(
                trainer.args, 
                False,
                trainer.accelerator,
                trainer.save_dtype,
                trainer.current_epoch.value,
                trainer.num_train_epochs,
                global_step,
                trainer.accelerator.unwrap_model(trainer.unet)
                )

            model_path = os.path.join(trainer.args.output_dir, ckpt_name)
            if copy_to_comfy_model_folder:
                shutil.copy(model_path, os.path.join(folder_paths.models_dir, "diffusion_models", "flux_trainer", ckpt_name))
                model_path = os.path.join(folder_paths.models_dir, "diffusion_models", "flux_trainer", ckpt_name)
            if end_training:
                trainer.accelerator.end_training()
        
        return (network_trainer, model_path, global_step)
    
class FluxTrainEnd:
    """* 参数详细说明
network_trainer:

类型: NETWORKTRAINER

说明: 网络训练器对象，包含训练器和训练循环的对象。

save_state:

类型: BOOLEAN

默认值: True

说明: 是否保存整个模型状态，包括模型的权重和其他相关信息。

* 内部逻辑
1. 初始化训练环境:
获取训练器和训练循环对象。

2. 更新元数据:
更新训练器元数据，包括当前训练周期和训练结束时间。

3. 获取LoRA模型:
获取训练器中的LoRA模型。

4. 结束训练:
结束训练过程。

5. 保存模型状态:
如果 save_state 为 True，保存整个模型状态。

6. 保存最终模型:
生成最终模型的检查点名称。
保存最终模型的权重。

7. 返回LoRA名称、元数据和LoRA路径:
返回最终LoRA模型的名称、元数据和LoRA路径。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": True}),
             },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("lora_name", "metadata", "lora_path",)
    FUNCTION = "endtrain"
    CATEGORY = "FluxTrainer"
    OUTPUT_NODE = True

    def endtrain(self, network_trainer, save_state):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]
            
            network_trainer.metadata["ss_epoch"] = str(network_trainer.num_train_epochs)
            network_trainer.metadata["ss_training_finished_at"] = str(time.time())

            network = network_trainer.accelerator.unwrap_model(network_trainer.network)

            network_trainer.accelerator.end_training()
            network_trainer.optimizer_eval_fn()

            if save_state:
                train_util.save_state_on_train_end(network_trainer.args, network_trainer.accelerator)

            ckpt_name = train_util.get_last_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as)
            network_trainer.save_model(ckpt_name, network, network_trainer.global_step, network_trainer.num_train_epochs, force_sync_upload=True)
            logger.info("model saved.")

            final_lora_name = str(network_trainer.args.output_name)
            final_lora_path = os.path.join(network_trainer.args.output_dir, ckpt_name)

            # metadata
            metadata = json.dumps(network_trainer.metadata, indent=2)

            training_loop = None
            network_trainer = None
            mm.soft_empty_cache()
            
        return (final_lora_name, metadata, final_lora_path)

class FluxTrainResume:
    """* 参数详细说明
load_state_path:

类型: STRING

默认值: ""

说明: 加载状态的路径，用于指定从哪个路径加载训练状态。

skip_until_initial_step:

类型: BOOLEAN

默认值: False

说明: 是否跳过直到初始步数，如果为 True，则在加载状态后跳过直到初始步数的训练。

* 内部逻辑
配置恢复训练的参数:

创建一个包含恢复训练参数的字典。

返回恢复训练的参数:

返回包含恢复训练参数的字典。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "load_state_path": ("STRING", {"default": "", "multiline": True, "tooltip": "path to load state from"}),
            "skip_until_initial_step" : ("BOOLEAN", {"default": False}),
             },
        }

    RETURN_TYPES = ("ARGS", )
    RETURN_NAMES = ("resume_args", )
    FUNCTION = "resume"
    CATEGORY = "FluxTrainer"

    def resume(self, load_state_path, skip_until_initial_step):
        resume_args ={
            "resume": load_state_path,
            "skip_until_initial_step": skip_until_initial_step
        }
            
        return (resume_args, )
    
class FluxTrainBlockSelect:
    """
参数详细说明
include:

类型: STRING

默认值: "lora_unet_single_blocks_20_linear2"

说明: 包含在LoRA网络中的块，用于指定哪些块将被用于LoRA训练。可以输入多个块，用逗号分隔。

* 内部逻辑
1. 解析输入字符串:
将输入字符串按逗号分割，处理每个块的名称。

2. 生成包含块的参数:
创建一个包含块的参数字典。

3. 返回包含块的参数:
返回包含块的参数字典。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "include": ("STRING", {"default": "lora_unet_single_blocks_20_linear2", "multiline": True, "tooltip": "blocks to include in the LoRA network, to select multiple blocks either input them as "}),
             },
        }

    RETURN_TYPES = ("ARGS", )
    RETURN_NAMES = ("block_args", )
    FUNCTION = "block_select"
    CATEGORY = "FluxTrainer"

    def block_select(self, include):
        import re
    
        # Split the input string by commas to handle multiple ranges/blocks
        elements = include.split(',')
    
        # Initialize a list to collect block names
        blocks = []
    
        # Pattern to find ranges like (10-20)
        pattern = re.compile(r'\((\d+)-(\d+)\)')
    
        # Extract the prefix and suffix from the first element
        prefix_suffix_pattern = re.compile(r'(.*)_blocks_(.*)')
    
        for element in elements:
            element = element.strip()
            match = prefix_suffix_pattern.match(element)
            if match:
                prefix = match.group(1) + "_blocks_"
                suffix = match.group(2)
                matches = pattern.findall(suffix)
                if matches:
                    for start, end in matches:
                        # Generate block names for the range and add them to the list
                        blocks.extend([f"{prefix}{i}{suffix.replace(f'({start}-{end})', '', 1)}" for i in range(int(start), int(end) + 1)])
                else:
                    # If no range is found, add the block name directly
                    blocks.append(element)
            else:
                blocks.append(element)
    
        # Construct the `include` string
        include_string = ','.join(blocks)
    
        block_args = {
            "include": f"only_if_contains={include_string}",
        }
    
        return (block_args, )
    
class FluxTrainValidationSettings:
    """
    FluxTrainValidationSettings 类提供了一个简单的接口，用于配置验证设置。通过指定采样步数、图像宽度、图像高度、指导缩放、种子、是否偏移调度以及基础偏移和最大偏移，用户可以灵活地进行验证设置。
#### 输入参数
- `steps`: 采样步骤数，默认值为20，范围从1到256。
- `width`: 图像宽度，默认值为512，范围从64到4096，步长为8。
- `height`: 图像高度，默认值为512，范围从64到4096，步长为8。
- `guidance_scale`: 指导尺度，默认值为3.5，范围从1.0到32.0，步长为0.05。
- `seed`: 随机种子，默认值为42。
- `shift`: 是否调整时间步长以有利于高时间步长的信号图像，默认值为True。
- `base_shift`: 基础偏移量，默认值为0.5。
- `max_shift`: 最大偏移量，默认值为1.15。

* 内部逻辑
创建验证设置字典:
创建一个包含验证设置的字典。

返回验证设置:
返回包含验证设置的字典。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 20, "min": 1, "max": 256, "step": 1, "tooltip": "sampling steps"}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image width"}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image height"}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 32.0, "step": 0.05, "tooltip": "guidance scale"}),
            "seed": ("INT", {"default": 42,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "shift": ("BOOLEAN", {"default": True, "tooltip": "shift the schedule to favor high timesteps for higher signal images"}),
            "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
            "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("VALSETTINGS", )
    RETURN_NAMES = ("validation_settings", )
    FUNCTION = "set"
    CATEGORY = "FluxTrainer"

    def set(self, **kwargs):
        validation_settings = kwargs
        print(validation_settings)

        return (validation_settings,)
        
class FluxTrainValidate:
    """- **输入**:
  - `network_trainer`: 包含网络训练器和训练循环的字典对象。
  - `validation_settings`: 可选参数，包含验证设置信息，如采样步数、图像尺寸等。
- **内部逻辑**:
  - 获取训练器和训练循环对象。
  - 使用给定的验证设置参数执行模型验证过程。
  - 使用加速器（accelerator）进行前向传播以生成图像数据。
  - 将生成的图像数据转换为适于显示的格式并返回。
- **输出**: 更新后的训练器对象及生成的验证图像。"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
            },
            "optional": {
                "validation_settings": ("VALSETTINGS",),
            }
        }

    RETURN_TYPES = ("NETWORKTRAINER", "IMAGE",)
    RETURN_NAMES = ("network_trainer", "validation_images",)
    FUNCTION = "validate"
    CATEGORY = "FluxTrainer"

    def validate(self, network_trainer, validation_settings=None):
        training_loop = network_trainer["training_loop"]
        network_trainer = network_trainer["network_trainer"]

        params = (
            network_trainer.accelerator, 
            network_trainer.args, 
            network_trainer.current_epoch.value, 
            network_trainer.global_step,
            network_trainer.unet,
            network_trainer.vae,
            network_trainer.text_encoder,
            network_trainer.sample_prompts_te_outputs,
            validation_settings
        )
        network_trainer.optimizer_eval_fn()
        image_tensors = network_trainer.sample_images(*params)

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, (0.5 * (image_tensors + 1.0)).cpu().float(),)
    
class VisualizeLoss:
    """
    - **输入**:
  - `network_trainer`: 包含网络训练器和训练循环的字典对象。
  - `plot_style`: Matplotlib绘图风格。
  - `window_size`: 移动平均窗口大小。
  - `normalize_y`: 是否将y轴归一化到0。
  - `width` 和 `height`: 绘图的宽度和高度（以像素为单位）。
  - `log_scale`: 是否使用对数比例尺绘制y轴。

- **内部逻辑**:
  - 获取网络训练器中的损失记录列表。
  - 应用移动平均处理损失值，以便平滑曲线。
  - 使用Matplotlib库创建并配置绘图，包括设置标题、标签、样式等。
  - 将生成的图像转换为张量格式，并返回给用户。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "plot_style": (plt.style.available,{"default": 'default', "tooltip": "matplotlib plot style"}),
            "window_size": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 1, "tooltip": "the window size of the moving average"}),
            "normalize_y": ("BOOLEAN", {"default": True, "tooltip": "normalize the y-axis to 0"}),
            "width": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 2, "tooltip": "width of the plot in pixels"}),
            "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 2, "tooltip": "height of the plot in pixels"}),
            "log_scale": ("BOOLEAN", {"default": False, "tooltip": "use log scale on the y-axis"}),
             },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("plot", "loss_list",)
    FUNCTION = "draw"
    CATEGORY = "FluxTrainer"

    def draw(self, network_trainer, window_size, plot_style, normalize_y, width, height, log_scale):
        import numpy as np
        loss_values = network_trainer["network_trainer"].loss_recorder.global_loss_list

        # Apply moving average
        def moving_average(values, window_size):
            return np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        if window_size > 0:
            loss_values = moving_average(loss_values, window_size)

        plt.style.use(plot_style)

        # Convert pixels to inches (assuming 100 pixels per inch)
        width_inches = width / 100
        height_inches = height / 100

        # Create a plot
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.plot(loss_values, label='Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        if normalize_y:
            plt.ylim(bottom=0)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image = Image.open(buf).convert('RGB')

        image_tensor = transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()

        return image_tensor, loss_values,

class FluxKohyaInferenceSampler:
    """`FluxKohyaInferenceSampler` 类用于从预训练模型和LoRA权重中生成图像。它支持应用或合并LoRA权重，并允许用户指定生成图像的各种参数，如步骤数、宽度、高度、引导尺度等。

#### 输入参数
- **flux_models**: 包含训练所需的模型路径的字典，包括transformer、vae、clip_l、t5和可选的lora_path。
- **lora_name**: LoRA的名称（从loras文件夹中选择）。
- **lora_method**: 是否应用或合并LoRA权重（选项为 "apply" 或 "merge"）。
- **steps**: 采样步骤数。
- **width**: 图像宽度。
- **height**: 图像高度。
- **guidance_scale**: 引导尺度。
- **seed**: 随机种子。
- **use_fp8**: 是否使用fp8权重（布尔值）。
- **apply_t5_attn_mask**: 是否应用t5注意力掩码（布尔值）。
- **prompt**: 提示文本。

#### 返回值
- 图像数据：生成的图像张量。

#### 内部逻辑

1. 初始化设备和数据类型：
   - 根据是否使用fp8来设置加速器和数据类型。

2. 加载预训练模型组件：
   - 加载CLIP-L模型和T5XXL模型，并进行评估模式设置。

3. 处理提示文本并编码：
   - 使用给定的提示文本进行标记化和编码处理，得到CLIP-L和T5XXL的输出。

4. 准备噪声张量：
   - 根据指定的高度和宽度准备噪声张量，并设置随机种子。

5. 定义调度函数：
   - 定义用于时间步长变换的时间偏移函数`time_shift`以及线性估计函数`get_lin_function`，并基于这些函数构建调度时间步长列表`get_schedule`。

6. 执行去噪操作：
   - 使用定义好的调度时间步长列表执行去噪操作，逐步更新噪声张量以生成最终图像。

7. 解码生成的图像：
   - 将经过去噪操作后的噪声张量解码为实际图像，并将其限制在[-1, 1]范围内以确保正确显示效果。
"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "flux_models": ("TRAIN_FLUX_MODELS",),
            "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
            "lora_method": (["apply", "merge"], {"tooltip": "whether to apply or merge the lora weights"}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 256, "step": 1, "tooltip": "sampling steps"}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image width"}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "image height"}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 32.0, "step": 0.05, "tooltip": "guidance scale"}),
            "seed": ("INT", {"default": 42,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "use_fp8": ("BOOLEAN", {"default": True, "tooltip": "use fp8 weights"}),
            "apply_t5_attn_mask": ("BOOLEAN", {"default": True, "tooltip": "use t5 attention mask"}),
            "prompt": ("STRING", {"multiline": True, "default": "illustration of a kitten", "tooltip": "prompt"}),
          
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "sample"
    CATEGORY = "FluxTrainer"

    def sample(self, flux_models, lora_name, steps, width, height, guidance_scale, seed, prompt, use_fp8, lora_method, apply_t5_attn_mask):

        from .library import flux_utils as flux_utils
        from .library import strategy_flux as strategy_flux
        from .networks import lora_flux as lora_flux
        from typing import List, Optional, Callable
        from tqdm import tqdm
        import einops
        import math
        import accelerate
        import gc

        device = "cuda"
        

        if use_fp8:
            accelerator = accelerate.Accelerator(mixed_precision="bf16")
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.float16
            accelerator = None
        loading_device = "cpu"
        ae_dtype = torch.bfloat16

        pretrained_model_name_or_path = flux_models["transformer"]
        clip_l = flux_models["clip_l"]
        t5xxl = flux_models["t5"]
        ae = flux_models["vae"]
        lora_path = folder_paths.get_full_path("loras", lora_name)

        # load clip_l
        logger.info(f"Loading clip_l from {clip_l}...")
        clip_l = flux_utils.load_clip_l(clip_l, None, loading_device)
        clip_l.eval()

        logger.info(f"Loading t5xxl from {t5xxl}...")
        t5xxl = flux_utils.load_t5xxl(t5xxl, None, loading_device)
        t5xxl.eval()

        if use_fp8:
            clip_l = accelerator.prepare(clip_l)
            t5xxl = accelerator.prepare(t5xxl)

        t5xxl_max_length = 512
        tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
        encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

        # DiT
        model = flux_utils.load_flow_model("dev", pretrained_model_name_or_path, dtype, loading_device)
        model.eval()
        logger.info(f"Casting model to {dtype}")
        model.to(dtype)  # make sure model is dtype
        if use_fp8:
            model = accelerator.prepare(model)

        # AE
        ae = flux_utils.load_ae("dev", ae, ae_dtype, loading_device)
        ae.eval()


        # LoRA
        lora_models: List[lora_flux.LoRANetwork] = []
        multiplier = 1.0

        lora_model, weights_sd = lora_flux.create_network_from_weights(
            multiplier, lora_path, ae, [clip_l, t5xxl], model, None, True
        )
        if lora_method == "merge":
            lora_model.merge_to([clip_l, t5xxl], model, weights_sd)
        elif lora_method == "apply":
            lora_model.apply_to([clip_l, t5xxl], model)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {lora_name}: {info}")
            lora_model.eval()
            lora_model.to(device)
        lora_models.append(lora_model)


        packed_latent_height, packed_latent_width = math.ceil(height / 16), math.ceil(width / 16)
        noise = torch.randn(
            1,
            packed_latent_height * packed_latent_width,
            16 * 2 * 2,
            device=device,
            dtype=ae_dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)

        # prepare embeddings
        logger.info("Encoding prompts...")
        tokens_and_masks = tokenize_strategy.tokenize(prompt)
        clip_l = clip_l.to(device)
        t5xxl = t5xxl.to(device)
        with torch.no_grad():
            if use_fp8:
                clip_l.to(ae_dtype)
                t5xxl.to(ae_dtype)
                with accelerator.autocast():
                    l_pooled, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, apply_t5_attn_mask
                    )
        # NaN check
        if torch.isnan(l_pooled).any():
            raise ValueError("NaN in l_pooled")
                
        if torch.isnan(t5_out).any():
            raise ValueError("NaN in t5_out")

        
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
      
        gc.collect()
        torch.cuda.empty_cache()

        # generate image
        logger.info("Generating image...")
        model = model.to(device)
        print("MODEL DTYPE: ", model.dtype)

        img_ids = img_ids.to(device)
        t5_attn_mask = t5_attn_mask.to(device) if apply_t5_attn_mask else None
        def time_shift(mu: float, sigma: float, t: torch.Tensor):
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


        def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b


        def get_schedule(
            num_steps: int,
            image_seq_len: int,
            base_shift: float = 0.5,
            max_shift: float = 1.15,
            shift: bool = True,
        ) -> list[float]:
            # extra step for zero
            timesteps = torch.linspace(1, 0, num_steps + 1)

            # shifting the schedule to favor high timesteps for higher signal images
            if shift:
                # eastimate mu based on linear estimation between two points
                mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
                timesteps = time_shift(mu, 1.0, timesteps)

            return timesteps.tolist()


        def denoise(
            model,
            img: torch.Tensor,
            img_ids: torch.Tensor,
            txt: torch.Tensor,
            txt_ids: torch.Tensor,
            vec: torch.Tensor,
            timesteps: list[float],
            guidance: float = 4.0,
            t5_attn_mask: Optional[torch.Tensor] = None,
        ):
            # this is ignored for schnell
            guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
            comfy_pbar = comfy.utils.ProgressBar(total=len(timesteps))
            for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
                t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    txt_attention_mask=t5_attn_mask,
                )
                img = img + (t_prev - t_curr) * pred
                comfy_pbar.update(1)

            return img
        def do_sample(
            accelerator: Optional[accelerate.Accelerator],
            model,
            img: torch.Tensor,
            img_ids: torch.Tensor,
            l_pooled: torch.Tensor,
            t5_out: torch.Tensor,
            txt_ids: torch.Tensor,
            num_steps: int,
            guidance: float,
            t5_attn_mask: Optional[torch.Tensor],
            is_schnell: bool,
            device: torch.device,
            flux_dtype: torch.dtype,
        ):
            timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)
            print(timesteps)

            # denoise initial noise
            if accelerator:
                with accelerator.autocast(), torch.no_grad():
                    x = denoise(
                        model, img, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=guidance, t5_attn_mask=t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=flux_dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
                with torch.autocast(device_type=device.type, dtype=flux_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, apply_t5_attn_mask
                    )

            return x
        
        x = do_sample(accelerator, model, noise, img_ids, l_pooled, t5_out, txt_ids, steps, guidance_scale, t5_attn_mask, False, device, dtype)
        
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # unpack
        x = x.float()
        x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

        # decode
        logger.info("Decoding image...")
        ae = ae.to(device)
        with torch.no_grad():
            if use_fp8:
                with accelerator.autocast():
                    x = ae.decode(x)
            else:
                with torch.autocast(device_type=device.type, dtype=ae_dtype):
                    x = ae.decode(x)

        ae = ae.cpu()

        x = x.clamp(-1, 1)
        x = x.permute(0, 2, 3, 1)

        return ((0.5 * (x + 1.0)).cpu().float(),)   

class UploadToHuggingFace:
    """
- **功能**: 将训练好的模型上传到Hugging Face模型库。
- **输入**:
  - `network_trainer`: 包含网络训练器和训练循环的字典对象。
  - `source_path`: 模型文件或文件夹的路径。
  - `repo_id`: Hugging Face仓库的ID。
  - `revision`: 仓库版本。
  - `private`: 是否创建私有仓库（布尔值）。
  - `token` (可选): Hugging Face API令牌，用于身份验证。"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "source_path": ("STRING", {"default": ""}),
                "repo_id": ("STRING",{"default": ""}),
                "revision": ("STRING", {"default": ""}),
                "private": ("BOOLEAN", {"default": True, "tooltip": "If creating a new repo, leave it private"}),
             },
             "optional": {
                "token": ("STRING", {"default": "","tooltip":"DO NOT LEAVE IN THE NODE or it might save in metadata, can also use the hf_token.json"}),
             }
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING",)
    RETURN_NAMES = ("network_trainer","status",)
    FUNCTION = "upload"
    CATEGORY = "FluxTrainer"

    def upload(self, source_path, network_trainer, repo_id, private, revision, token=""):
        """- **输入**:
  - network_trainer, source_path, repo_id, private, revision, token
- **内部逻辑**:
  1. 初始化Hugging Face API并读取令牌（如果未提供，则从本地文件中读取）。
  2. 检查指定的仓库是否已存在。如果不存在，则创建一个新的仓库（根据`private`参数设置为私有或公开）。
  3. 将模型文件或整个目录上传到指定的Hugging Face仓库中。同时上传包含元数据信息的JSON文件。
- **输出**: 更新后的网络训练器对象和上传状态字符串。"""
        with torch.inference_mode(False):
            from huggingface_hub import HfApi
            
            if not token:
                with open(os.path.join(script_directory, "hf_token.json"), "r") as file:
                    token_data = json.load(file)
                token = token_data["hf_token"]
            print(token)

            # Save metadata to a JSON file
            directory_path = os.path.dirname(os.path.dirname(source_path))
            file_name = os.path.basename(source_path)

            metadata = network_trainer["network_trainer"].metadata
            metadata_file_path = os.path.join(directory_path, "metadata.json")
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            repo_type = None
            api = HfApi(token=token)

            try:
                api.repo_info(
                    repo_id=repo_id, 
                    revision=revision if revision != "" else None, 
                    repo_type=repo_type)
                repo_exists = True
                logger.info(f"Repository {repo_id} exists.")
            except Exception as e:  # Catching a more specific exception would be better if you know what to expect
                repo_exists = False
                logger.error(f"Repository {repo_id} does not exist. Exception: {e}")
            
            if not repo_exists:
                try:
                    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
                except Exception as e:  # Checked for RepositoryNotFoundError, but other exceptions could be problematic
                    logger.error("===========================================")
                    logger.error(f"failed to create HuggingFace repo: {e}")
                    logger.error("===========================================")

            is_folder = (type(source_path) == str and os.path.isdir(source_path)) or (isinstance(source_path, Path) and source_path.is_dir())
            print(source_path, is_folder)

            try:
                if is_folder:
                    api.upload_folder(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        folder_path=source_path,
                        path_in_repo=file_name,
                    )
                else:
                    api.upload_file(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        path_or_fileobj=source_path,
                        path_in_repo=file_name,
                    )
                # Upload the metadata file separately if it's not a folder upload
                if not is_folder:
                    api.upload_file(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        path_or_fileobj=str(metadata_file_path),
                        path_in_repo='metadata.json',
                    )
                status = "Uploaded to HuggingFace succesfully"
            except Exception as e:  # RuntimeErrorを確認済みだが他にあると困るので
                logger.error("===========================================")
                logger.error(f"failed to upload to HuggingFace / HuggingFaceへのアップロードに失敗しました : {e}")
                logger.error("===========================================")
                status = f"Failed to upload to HuggingFace {e}"
                
            return (network_trainer, status,)
        
class ExtractFluxLoRA:
    """#### 功能
该类的主要功能是从两个模型中提取LoRA权重，并将其保存到指定路径。具体来说，它从一个原始模型和一个微调后的模型中提取LoRA权重，并将其保存为指定格式的文件。

#### 输入参数
- `original_model`: 原始模型文件路径。
- `finetuned_model`: 微调后的模型文件路径。
- `output_path`: 输出路径，用于存储提取的LoRA权重文件。
- `dim`: LoRA维度（rank）。
- `save_dtype`: 保存LoRA权重的数据类型，可以是`fp32`, `fp16`, `bf16`, `fp8_e4m3fn`, 或`fp8_e5m2`。
- `load_device`: 加载模型时使用的设备（`cpu` 或`cuda`）。
- `store_device`: 存储LoRA权重时使用的设备（`cpu` 或`cuda`）。
- `clamp_quantile`: 量化百分位数，用于限制权重范围。
- `metadata`: 是否构建元数据信息。
- `mem_eff_safe_open`: 是否使用内存高效的加载方式。

#### 输出
返回包含输出路径的字符串。
"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_model": (folder_paths.get_filename_list("unet"), ),
                "finetuned_model": (folder_paths.get_filename_list("unet"), ),
                "output_path": ("STRING", {"default": f"{str(os.path.join(folder_paths.models_dir, 'loras', 'Flux'))}"}),
                "dim": ("INT", {"default": 4, "min": 2, "max": 1024, "step": 2, "tooltip": "LoRA rank"}),
                "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "bf16", "tooltip": "the dtype to save the LoRA as"}),
                "load_device": (["cpu", "cuda"], {"default": "cuda", "tooltip": "the device to load the model to"}),
                "store_device": (["cpu", "cuda"], {"default": "cpu", "tooltip": "the device to store the LoRA as"}),
                "clamp_quantile": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "clamp quantile"}),
                "metadata": ("BOOLEAN", {"default": True, "tooltip": "build metadata"}),
                "mem_eff_safe_open": ("BOOLEAN", {"default": False, "tooltip": "memory efficient loading"}),
             },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("output_path",)
    FUNCTION = "extract"
    CATEGORY = "FluxTrainer"

    def extract(self, original_model, finetuned_model, output_path, dim, save_dtype, load_device, store_device, clamp_quantile, metadata, mem_eff_safe_open):
        """#### 详细说明

1. **输入参数处理**：
   - 获取原始模型和微调后模型的完整路径。
   - 设置输出文件名，基于微调后模型名称，并加上提取的LoRA权重信息。

2. **核心逻辑**：
   - 使用从自定义模块导入的`sparse_utils.svd()`函数来执行SVD分解以提取LoRA权重。该函数需要多个参数，包括原始和微调后的模型路径、输出文件路径、维度、设备类型、存储设备类型等。

3. **返回结果**：
   - 返回一个包含输出路径的元组。"""
        from .flux_extract_lora import svd
        transformer_path = folder_paths.get_full_path("unet", original_model)
        finetuned_model_path = folder_paths.get_full_path("unet", finetuned_model)
        outpath = svd(
            model_org = transformer_path,
            model_tuned = finetuned_model_path,
            save_to = os.path.join(output_path, f"{finetuned_model.replace('.safetensors', '')}_extracted_lora_rank_{dim}-{save_dtype}.safetensors"),
            dim = dim,
            device = load_device,
            store_device = store_device,
            save_precision = save_dtype,
            clamp_quantile = clamp_quantile,
            no_metadata = not metadata,
            mem_eff_safe_open = mem_eff_safe_open
        )
     
        return (outpath,)

NODE_CLASS_MAPPINGS = {
    "InitFluxLoRATraining": InitFluxLoRATraining,
    "InitFluxTraining": InitFluxTraining,
    "FluxTrainModelSelect": FluxTrainModelSelect,
    "TrainDatasetGeneralConfig": TrainDatasetGeneralConfig,
    "TrainDatasetAdd": TrainDatasetAdd,
    "FluxTrainLoop": FluxTrainLoop,
    "VisualizeLoss": VisualizeLoss,
    "FluxTrainValidate": FluxTrainValidate,
    "FluxTrainValidationSettings": FluxTrainValidationSettings,
    "FluxTrainEnd": FluxTrainEnd,
    "FluxTrainSave": FluxTrainSave,
    "FluxKohyaInferenceSampler": FluxKohyaInferenceSampler,
    "UploadToHuggingFace": UploadToHuggingFace,
    "OptimizerConfig": OptimizerConfig,
    "OptimizerConfigAdafactor": OptimizerConfigAdafactor,
    "FluxTrainSaveModel": FluxTrainSaveModel,
    "ExtractFluxLoRA": ExtractFluxLoRA,
    "OptimizerConfigProdigy": OptimizerConfigProdigy,
    "FluxTrainResume": FluxTrainResume,
    "FluxTrainBlockSelect": FluxTrainBlockSelect,
    "TrainDatasetRegularization": TrainDatasetRegularization,
    "FluxTrainAndValidateLoop": FluxTrainAndValidateLoop,
    "OptimizerConfigProdigyPlusScheduleFree": OptimizerConfigProdigyPlusScheduleFree,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InitFluxLoRATraining": "Init Flux LoRA Training",
    "InitFluxTraining": "Init Flux Training",
    "FluxTrainModelSelect": "FluxTrain ModelSelect",
    "TrainDatasetGeneralConfig": "TrainDatasetGeneralConfig",
    "TrainDatasetAdd": "TrainDatasetAdd",
    "FluxTrainLoop": "Flux Train Loop",
    "VisualizeLoss": "Visualize Loss",
    "FluxTrainValidate": "Flux Train Validate",
    "FluxTrainValidationSettings": "Flux Train Validation Settings",
    "FluxTrainEnd": "Flux LoRA Train End",
    "FluxTrainSave": "Flux Train Save LoRA",
    "FluxKohyaInferenceSampler": "Flux Kohya Inference Sampler",
    "UploadToHuggingFace": "Upload To HuggingFace",
    "OptimizerConfig": "Optimizer Config",
    "OptimizerConfigAdafactor": "Optimizer Config Adafactor",
    "FluxTrainSaveModel": "Flux Train Save Model",
    "ExtractFluxLoRA": "Extract Flux LoRA",
    "OptimizerConfigProdigy": "Optimizer Config Prodigy",
    "FluxTrainResume": "Flux Train Resume",
    "FluxTrainBlockSelect": "Flux Train Block Select",
    "TrainDatasetRegularization": "Train Dataset Regularization",
    "FluxTrainAndValidateLoop": "Flux Train And Validate Loop",
    "OptimizerConfigProdigyPlusScheduleFree": "Optimizer Config ProdigyPlusScheduleFree",
}
