"""该代码文件是一个用于训练神经网络的Python脚本，主要用于训练Stable Diffusion模型中的U-Net和Text Encoder部分。代码通过导入多个模块和库，定义了一个NetworkTrainer类，该类包含了训练过程中所需的各种方法，如模型加载、数据加载、优化器设置、训练循环等。代码还支持多种训练参数的配置，如学习率、批量大小、优化器类型等。"""
import importlib
import argparse
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
from typing import Any, List
import toml

from tqdm import tqdm
from comfy.utils import ProgressBar

import torch
from .library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from .library import deepspeed_utils, model_util, strategy_base, strategy_sd

from .library import train_util as train_util
from .library.train_util import DreamBoothDataset
from .library import config_util as config_util
from .library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from .library import huggingface_util as huggingface_util
from .library import custom_train_functions as custom_train_functions
from .library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from .library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


class NetworkTrainer:
    """NetworkTrainer是一个基类，是整个训练过程的核心，包含了初始化、模型加载、数据加载、训练循环、日志生成等方法。该类的主要功能是协调各个模块，确保训练过程顺利进行。"""
    def __init__(self):
        """功能: 初始化NetworkTrainer类的实例。

作用: 设置一些默认参数，如vae_scale_factor和is_sdxl。"""
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        """功能: 生成训练步骤的日志。

输入:

args: 命令行参数。

current_loss: 当前损失值。

avr_loss: 平均损失值。

lr_scheduler: 学习率调度器。

lr_descriptions: 学习率描述。

keys_scaled: 缩放的键。

mean_norm: 平均范数。

maximum_norm: 最大范数。

输出: 包含损失和学习率信息的日志字典。"""
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        """功能: 断言额外的参数。

作用: 检查并断言额外的训练参数。"""
        pass

    def load_target_model(self, args, weight_dtype, accelerator):
        """功能: 加载目标模型。

输入:

args: 命令行参数。

weight_dtype: 权重数据类型。

accelerator: 加速器。

输出: 加载的模型版本字符串、文本编码器、VAE和U-Net。"""
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        # Incorporate xformers or memory efficient attention into the model
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # If you have xformers compatible with PyTorch 2.0.0 or higher, you can use the following
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def get_tokenize_strategy(self, args):
        """功能: 获取分词策略。
输入: args: 命令行参数。
输出: 分词策略实例。
"""
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        """功能: 获取分词器。
输入: tokenize_strategy: 分词策略实例。
输出: 分词器列表。"""
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        """功能: 获取潜在变量缓存策略。
输入: args: 命令行参数。
输出: 潜在变量缓存策略实例。"""
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, False
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        """功能: 获取文本编码策略。
输入: args: 命令行参数。
输出: 文本编码策略实例。"""
        return strategy_sd.SdTextEncodingStrategy(args.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, args):
        """功能: 获取文本编码器输出缓存策略。
输入: args: 命令行参数。
输出: 文本编码器输出缓存策略实例。"""
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """功能: 获取用于文本编码的模型。

输入:

args: 命令行参数。

accelerator: 加速器。

text_encoders: 文本编码器列表。

输出: 文本编码器列表。
        Returns a list of models that will be used for text encoding. SDXL uses wrapped and unwrapped models.
        """
        return text_encoders

    # returns a list of bool values indicating whether each text encoder should be trained
    def get_text_encoders_train_flags(self, args, text_encoders):
        """功能: 获取文本编码器的训练标志。
输入:
args: 命令行参数。
text_encoders: 文本编码器列表。
输出: 文本编码器的训练标志列表。"""
        return [True] * len(text_encoders) if self.is_train_text_encoder(args) else [False] * len(text_encoders)

    def is_train_text_encoder(self, args):
        """功能: 判断是否训练文本编码器。
输入: args: 命令行参数。
输出: 布尔值，表示是否训练文本编码器。"""
        return not args.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        """功能: 如果需要，缓存文本编码器输出。
输入:
args: 命令行参数。
accelerator: 加速器。
unet: U-Net模型。
vae: VAE模型。
text_encoders: 文本编码器列表。
dataset: 数据集。
weight_dtype: 权重数据类型。
作用: 缓存文本编码器的输出。"""
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        """功能: 调用U-Net模型。

输入:

args: 命令行参数。

accelerator: 加速器。

unet: U-Net模型。

noisy_latents: 噪声潜在变量。

timesteps: 时间步。

text_conds: 文本条件。

batch: 批次数据。

weight_dtype: 权重数据类型。

输出: 噪声预测。"""
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        """功能: 对网络进行全局归约。
输入:
accelerator: 加速器。
network: 网络模型。
作用: 对网络参数进行全局归约。"""
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        """功能: 采样图像。
输入:
accelerator: 加速器。
args: 命令行参数。
epoch: 当前轮数。
global_step: 全局步数。
device: 设备。
vae: VAE模型。
tokenizers: 分词器列表。
text_encoder: 文本编码器。
unet: U-Net模型。
作用: 采样图像。
"""
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    # region SD/SDXL

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        """功能: 对网络进行后处理。
输入:
args: 命令行参数。
accelerator: 加速器。
network: 网络模型。
text_encoders: 文本编码器列表。
unet: U-Net模型。
作用: 对网络进行后处理。"""
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        """功能: 获取噪声调度器。
输入:
args: 命令行参数。
device: 设备。
输出: 噪声调度器实例。
"""
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, accelerator, vae, images):
        """功能: 将图像编码为潜在变量。
输入:
args: 命令行参数。
accelerator: 加速器。
vae: VAE模型。
images: 图像。
输出: 潜在变量。"""
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents):
        """功能: 缩放潜在变量。
输入:
args: 命令行参数。
latents: 潜在变量。
输出: 缩放后的潜在变量。"""
        return latents * self.vae_scale_factor

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
    ):
        """功能: 获取噪声预测和目标。
输入:
args: 命令行参数。
accelerator: 加速器。
noise_scheduler: 噪声调度器。
latents: 潜在变量。
batch: 批次数据。
text_encoder_conds: 文本编码器条件。
unet: U-Net模型。
network: 网络模型。
weight_dtype: 权重数据类型。
train_unet: 是否训练U-Net。
输出: 噪声预测、目标、时间步、Huber常数、权重。"""
        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        # Predict the noise residual
        with accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        if args.v_parameterization:
            # v-parameterization training
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        return noise_pred, target, timesteps, huber_c, None

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        """功能: 对损失进行后处理。
输入:
loss: 损失。
args: 命令行参数。
timesteps: 时间步。
noise_scheduler: 噪声调度器。
输出: 后处理后的损失。"""
        if args.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
        if args.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if args.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
        if args.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)
        return loss

    def get_sai_model_spec(self, args):
        """功能: 获取SAI模型规格。
输入: args: 命令行参数。
输出: SAI模型规格。"""
        return train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)

    def update_metadata(self, metadata, args):
        """更新元数据"""
        pass

    def is_text_encoder_not_needed_for_training(self, args):
        """功能: 判断是否不需要文本编码器进行训练。
输入: args: 命令行参数。
输出: 布尔值，表示是否不需要文本编码器进行训练。"""
        return False  # use for sample images

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        """功能: 准备文本编码器的梯度检查点。
输入:
index: 索引。
text_encoder: 文本编码器。
作用: 准备文本编码器的梯度检查点。"""
        # set top parameter requires_grad = True for gradient checkpointing works
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        """功能: 准备文本编码器的FP8。
输入:
index: 索引。
text_encoder: 文本编码器。
te_weight_dtype: 文本编码器权重数据类型。
weight_dtype: 权重数据类型。
作用: 准备文本编码器的FP8。"""
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    # endregion

    def init_train(self, args):
        """功能: 初始化训练。
输入: args: 命令行参数。
作用: 初始化训练过程。"""
        """1. 初始化训练环境
        生成一个随机的 session_id 和记录训练开始时间。
验证训练参数，准备数据集参数，并准备 DeepSpeed 相关的参数。
PS:DeepSpeed是由微软开发的一个开源深度学习优化库
，它旨在提高大规模模型训练的效率和可扩展性。DeepSpeed通过创新的算法和技术，如高效的并行化策略、内存优化技术（ZeRO）、混合精度训练等，显著提升了训练速度并降低了资源需求
。它支持多种并行方法，包括数据并行、模型并行和流水线并行，并与PyTorch等主流框架无缝集成，
设置日志记录。
检查是否需要缓存潜在变量 (cache_latents)。
判断是否使用 DreamBooth 方法 (use_dreambooth_method) 或用户配置 (use_user_config)。
如果未指定种子 (seed)，则生成一个随机种子并设置。"""
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)
        """2. 获取分词策略和分词器
        获取并设置分词器列表（尽管它将在未来的重构中被移除）,设置潜在变量缓存策略，并将其作为全局策略设置。"""
        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        """3. 获取潜在变量缓存策略
        获取潜在变量缓存策略 (latents_caching_strategy) 并设置为当前策略。"""
        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)


        """4. 准备数据集
        初始化进度条 (pbar)。

        如果没有指定数据集类 (dataset_class)，则根据用户配置或 DreamBooth 方法生成数据集配置 (user_config)。

        生成数据集蓝图 (blueprint) 并根据蓝图生成数据集组 (train_dataset_group)。

        初始化当前轮数 (current_epoch) 和当前步数 (current_step)。

        初始化数据加载器 (collator)。

        如果启用了调试数据集 (debug_dataset)，则设置当前策略并调试数据集。

        检查数据集是否为空，如果为空则报错并返回。

        如果需要缓存潜在变量 (cache_latents)，则检查数据集是否可缓存。

        断言额外的参数 (assert_extra_args)。"""
        pbar = ProgressBar(5)

        # Prepare the dataset
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dasaset needs to know the strategies explicitly
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group)  # may change some args


        """5. 准备加速器
        准备加速器 (accelerator)。初始化加速器，为后续的训练过程提供加速支持。"""
        # prepare accelerator
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)

        """6. 准备数据类型和模型
        准备权重数据类型 (weight_dtype) 和保存数据类型 (save_dtype)。
        根据是否禁用半精度 VAE (no_half_vae) 设置 VAE 数据类型 (vae_dtype)。
        加载目标模型 (load_target_model)，获取模型版本 (model_version)、文本编码器 (text_encoder)、VAE (vae) 和 U-Net (unet)。
        将文本编码器 (text_encoder) 转换为列表 (text_encoders)。
        更新进度条 (pbar)。
        """
        # Prepare a type that supports mixed precision and cast it as appropriate.
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # Load the model
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        pbar.update(1)

        # Load the model for incremental learning
        #sys.path.append(os.path.dirname(__file__))
        """导入网络模块：
        打印要导入的网络模块名称 (args.network_module)。
        获取当前包的名称 (package)。
        使用 importlib.import_module 动态导入指定的网络模块 (network_module)。"""
        accelerator.print("import network module:", args.network_module)
        package = __name__.split('.')[0]
        network_module = importlib.import_module(args.network_module, package=package)

        """检查并加载基础权重：
        - **运行逻辑**:
        - 检查是否指定了基础权重 (`args.base_weights`)。
        - 遍历每个基础权重路径，并确定对应的乘数 (`multiplier`)。如果未指定乘数，则默认为1.0。
        - 使用 `network_module.create_network_from_weights` 方法从指定路径加载网络权重，并创建一个临时网络模块 (`module`) 和状态字典 (`weights_sd`)。
        - 将加载的权重合并到文本编码器 (`text_encoder`) 和 U-Net (`unet`) 中。
        - **目的**:
        - 将预训练的基础权重加载到当前模型中，并通过乘数调整其影响。这有助于增量学习，即在现有模型基础上进行进一步训练。"""
        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        """7. 缓存潜在变量
        如果需要缓存潜在变量 (cache_latents)，则将 VAE 移动到加速器设备 (accelerator.device) 并设置为评估模式 (eval)。
        缓存潜在变量 (new_cache_latents)。
        将 VAE 移动回 CPU 并清理设备内存 (clean_memory_on_device)。
        """
        # cache latents
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, True)

            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

        """8. 缓存文本编码器输出
        获取文本编码策略 (text_encoding_strategy) 并设置为当前策略。
        获取文本编码器输出缓存策略 (text_encoder_outputs_caching_strategy) 并设置为当前策略（如果存在）。
        如果需要，缓存文本编码器输出 (cache_text_encoder_outputs_if_needed)。
        更新进度条 (pbar)。
        """
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
        if text_encoder_outputs_caching_strategy is not None:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)

        pbar.update(1)

        """9. 准备网络
        """
        """解析网络参数：
        初始化 net_kwargs 字典。
        如果指定了网络参数 (args.network_args)，则遍历每个参数，将其解析为键值对并存储在 net_kwargs 中。"""
        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        """- **运行逻辑**:
  - 如果 `args.dim_from_weights` 为真，则从预训练权重文件中创建网络。
  - 否则，根据命令行参数创建一个新的网络实例。
- **目的**:
- 根据用户指定的参数（或预训练权重）创建网络实例。如果未指定某些参数（如 `dropout`），则使用默认值。"""
        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        """- **运行逻辑**:
        - 检查是否成功创建了网络实例。如果没有，则返回。
        - 检查网络是否有设置乘数的方法 (`set_multiplier`)。
        - 如果网络有准备方法 (`prepare_network`)，则调用该方法进行初始化。
        - 如果用户指定了缩放权重范数 (`scale_weight_norms`) 并且当前网络不支持该功能，则记录警告信息并禁用该功能。
        - **目的**:
        - 确保所有必要的功能都已正确设置，并根据需要调整配置。"""
        if network is None:
            return
        network_has_multiplier = hasattr(network, "set_multiplier")

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        """后处理和应用到模型
        - **运行逻辑**:
            - 调用 `post_process_network` 方法进行后处理操作。
            - 根据命令行参数确定是否训练U-Net和文本编码器。
            - **目的**:
            - 对新创建的网络进行必要的后处理操作，并将其应用到U-Net和文本编码器中。"""
        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        """加载预训练权重
        - **运行逻辑**:
            - 如果指定了预训练权重文件路径 (`args.network_weights`)，则加载这些权重到当前网络中，并打印加载信息。
            - **目的**:
            - 加载预训练的权重以初始化或微调当前的模型结构。"""
        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        """启用梯度检查点
        - **运行逻辑**：
        - 如果启用了梯度检查点 (`gradient_checkpointing`)：
            * 对于 U-NET 和文本编码器，启用梯度检查点机制。如果指定了CPU卸载(`cpu_offload_checkpointing`)，则使用CPU卸载策略来节省显存空间。否则直接启用梯度检查点机制以减少内存消耗并加快计算速度。
        理论上讲直接开启梯度比卸载到cpu上快，因为直接梯度检查是在GPU上运行的
        * 删除临时变量`t_enc`以释放内存资源。
        * 如果当前使用的自定义模块支持梯度检查点，则启用该机制以进一步优化性能和内存使用情况。"""
        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders)):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # Prepare classes necessary for learning
        accelerator.print("prepare optimizer, data loader etc.")

        """10. 准备优化器和数据加载器
        检查网络是否支持多学习率 (support_multiple_lrs)。
        根据是否支持多学习率，设置 text_encoder_lr。
        调用 network.prepare_optimizer_params 或 network.prepare_optimizer_params_with_multiple_te_lrs 方法，获取可训练参数 (trainable_params) 和学习率描述 (lr_descriptions)。
        获取优化器名称 (optimizer_name)、优化器参数 (optimizer_args) 和优化器实例 (optimizer)。
        获取优化器的训练和评估函数 (optimizer_train_fn 和 optimizer_eval_fn)。
        """
        # make backward compatibility for text_encoder_lr
        support_multiple_lrs = hasattr(network, "prepare_optimizer_params_with_multiple_te_lrs")
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            text_encoder_lr = None if args.text_encoder_lr is None or len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, args.unet_lr, args.learning_rate)
            else:
                results = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr, args.learning_rate)
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError as e:
            trainable_params = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr)
            lr_descriptions = None

        # if len(trainable_params) == 0:
        #     accelerator.print("no trainable parameters found / 学習可能なパラメータが見つかりませんでした")
        # for params in trainable_params:
        #     for k, v in params.items():
        #         if type(v) == float:
        #             pass
        #         else:
        #             v = len(v)
        #         accelerator.print(f"trainable_params: {k} = {v}")
        """获取优化器名称 (optimizer_name)、优化器参数 (optimizer_args) 和优化器实例 (optimizer)。"""
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
        """获取优化器的训练和评估函数 (optimizer_train_fn 和 optimizer_eval_fn)。"""
        self.optimizer_train_fn, self.optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)


        """2. 准备数据加载器"""
        # prepare dataloader
        # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
        # some strategies can be None
        """设置当前策略 (train_dataset_group.set_current_strategies)。"""
        train_dataset_group.set_current_strategies()

        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        """计算数据加载器的进程数 (n_workers)。"""
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        """创建数据加载器 (train_dataloader)，设置批量大小、是否打乱数据、数据加载器的进程数和是否持久化工作进程。"""
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # # Calculate the number of learning steps
        # if args.max_train_epochs is not None:
        #     args.max_train_steps = args.max_train_epochs * math.ceil(
        #         len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        #     )
        #     accelerator.print(
        #         f"override steps. steps for {args.max_train_epochs} epochs is {args.max_train_steps}"
        #     )

        # Send learning steps to the dataset side as well
        """3. 计算学习步数"""
        """将最大训练步数 (args.max_train_steps) 发送给数据集组 (train_dataset_group.set_max_train_steps)。"""
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr scheduler init
        """初始化学习率调度器 (lr_scheduler)。"""
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        """4. 是否启用混合精度进行训练
        如果启用了全精度训练 (args.full_fp16 或 args.full_bf16)，则检查混合精度设置，并将网络转换为相应的数据类型 (weight_dtype)。"""
        # Experimental function: performs fp16/bf16 learning including gradients, sets the entire model to fp16/bf16
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)
        """5. 是否启用 FP8 训练
        如果启用了 FP8 训练 (args.fp8_base 或 args.fp8_base_unet)，则检查 PyTorch 版本和混合精度设置，并将 U-Net 和文本编码器转换为相应的 FP8 数据类型。"""
        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base or args.fp8_base_unet:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16'"
            accelerator.print("enable fp8 training for U-Net.")
            unet_weight_dtype = torch.float8_e4m3fn if args.fp8_dtype == "e4m3" else torch.float8_e5m2
            accelerator.print(f"unet_weight_dtype: {unet_weight_dtype}")

            if not args.fp8_base_unet and not args.network_train_unet_only:
                accelerator.print("enable fp8 training for Text Encoder.")
            te_weight_dtype = torch.float8_e4m3fn if args.fp8_dtype == "e4m3" else torch.float8_e5m2

            # unet.to(accelerator.device)  # this makes faster `to(dtype)` below, but consumes 23 GB VRAM
            # unet.to(dtype=unet_weight_dtype)  # without moving to gpu, this takes a lot of time and main memory
            
            unet.to(accelerator.device, dtype=unet_weight_dtype)  # this seems to be safer than above
        """6. 准备模型和优化器"""
        """设置 UNet 模型的所有参数不需要梯度计算。并将 UNet 模型的数据类型设置为指定的类型"""
        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)
        """- 遍历所有文本编码器 (`text_encoders`)：
        设置每个文本编码器的所有参数不需要梯度计算
        如果设备不是 CPU，则将文本编码器的数据类型设置为指定的类型 
        如果文本编码器的数据类型与默认权重数据类型不同，则调用特定方法处理 FP8 数据类型"""
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)

                # nn.Embedding not support FP8
                if te_weight_dtype != weight_dtype:
                    self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)
        
        """是否使用 DeepSpeed 准备模型
        使用 DeepSpeed 进行分布式训练，准备模型、优化器、数据加载器和学习率调度器，为后续的训练过程做准备"""
        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            """获取文本编码器训练标志：flags 是一个布尔列表，指示哪些文本编码器需要进行训练。"""
            flags = self.get_text_encoders_train_flags(args, text_encoders)
            """调用 deepspeed_utils.prepare_deepspeed_model 函数来准备 DeepSpeed 模型。
                unet：如果 train_unet 为真，则传递 UNET 模型，否则传递 None。
                text_encoder1：如果第一个文本编码器需要训练，则传递第一个文本编码器，否则传递 None。
                text_encoder2：如果有第二个文本编码器并且需要训练，则传递第二个文本编码器，否则传递 None。
                network：传递网络模型。"""
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoders[0] if flags[0] else None,
                text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
                network=network,
            )
            """调用 accelerator.prepare 方法，传入 DeepSpeed 模型 (ds_model)、优化器 (optimizer)、数据加载器 (train_dataloader) 和学习率调度器 (lr_scheduler)。使用加速器（通常是 PyTorch 的 DDP 或其他并行技术）来准备所有必要的组件。"""
            ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, lr_scheduler
            )
            """使用加速器（通常是 PyTorch 的 DDP 或其他并行技术）来准备所有必要的组件。"""
            training_model = ds_model
        else:
            """如果需要训练 UNET，则使用加速器准备 UNET。"""
            if train_unet:
                unet = accelerator.prepare(unet)
            else:
                """如果不需要训练 UNET，则将 UNET 移动到目标设备并设置数据类型。"""
                unet.to(accelerator.device, dtype=unet_weight_dtype)  # move to device because unet is not prepared by accelerator
            if train_text_encoder:
                """遍历文本编码器列表 (text_encoders)，根据训练标志 (flags) 决定是否调用 accelerator.prepare 方法准备文本编码器。"""
                text_encoders = [
                    (accelerator.prepare(t_enc) if flag else t_enc)
                    for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))
                ]
                """如果文本编码器列表长度大于 1，则将 text_encoder 设置为 text_encoders；否则设置为 text_encoders[0]。"""
                if len(text_encoders) > 1:
                    text_encoder = text_encoders
                else:
                    text_encoder = text_encoders[0]
            else:
                pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set
            """调用 accelerator.prepare 方法，传入网络 (network)、优化器 (optimizer)、数据加载器 (train_dataloader) 和学习率调度器 (lr_scheduler)。"""
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )
            """将准备好的网络赋值给 training_model。"""
            training_model = network

        """7. 是否启用梯度检查点"""
        if args.gradient_checkpointing:
            """如果启用了梯度检查点 (args.gradient_checkpointing)，则将 U-Net 和文本编码器设置为训练模式，并启用梯度检查点。"""
            # according to TI example in Diffusers, train is required
            unet.train()
            for i, (t_enc, frag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            """否则，将 U-Net 和文本编码器设置为评估模式。"""
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc
        """准备网络的梯度相关操作 (prepare_grad_etc)。"""
        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        """8. 准备 VAE"""
        if not cache_latents:  # If you do not cache, VAE will be used, so enable VAE preparation.
            """如果未缓存潜在变量 (cache_latents)，则将 VAE 设置为评估模式，用来训练过程中将像素空间转换成缓存变量"""
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        """9. 启用全精度训练补丁
        如果启用了全精度训练 (args.full_fp16)，则应用补丁以启用 FP16 训练中的梯度缩放。"""
        # Experimental feature: Perform fp16 learning including gradients Apply a patch to PyTorch to enable grad scale in fp16
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        pbar.update(1)

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            """在保存模型时，仅保存网络模型的权重，并记录当前的训练状态（轮数和步数）。
            移除非网络模型的权重：
                遍历 models 列表，找出不属于网络模型的索引 (remove_indices)。
                从 weights 列表中移除这些索引对应的权重。"""
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            #if args.deepspeed:
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)
            # print(f"save model hook: {len(weights)} weights will be saved")

            """保存当前轮数和步数：
                构建训练状态文件路径 (train_state_file)。
                记录当前轮数 (current_epoch.value) 和步数 (current_step.value + 1)，并将其保存到 train_state_file 文件中。"""
            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            """在加载模型时，仅加载网络模型，并恢复当前的训练状态（轮数和步数）。"""
            # remove models except network
            """移除非网络模型：
            遍历 models 列表，找出不属于网络模型的索引 (remove_indices)。
            从 models 列表中移除这些索引对应的模型。"""
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")
            """加载当前轮数和步数：
            构建训练状态文件路径 (train_state_file)。
            如果 train_state_file 文件存在，则读取文件内容，获取当前步数 (steps_from_state)，并记录日志。"""
            # load current epoch and step to
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        # 注册钩子
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # 从huggingface上恢复
        # resume from local or huggingface
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        pbar.update(1)

        """1. 计算每轮的更新步数和总轮数"""
        # Calculate the number of epochs
        # 计算每轮的更新步数 (num_update_steps_per_epoch)，即数据加载器的长度除以梯度累积步数。
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        # 计算总轮数 (num_train_epochs)，即最大训练步数除以每轮的更新步数。
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        # 如果指定了 save_n_epoch_ratio，则计算每隔多少轮保存一次模型 (args.save_every_n_epochs)。
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        """2. 打印训练信息"""
        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        # 计算总批量大小 (total_batch_size)，即每个设备的批量大小乘以进程数和梯度累积步数。
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        # 打印训练信息，包括训练图像数量、正则化图像数量、每轮的批次数、总轮数、每个设备的批量大小、梯度累积步数和总优化步数。
        accelerator.print("running training")
        accelerator.print(f"  num train images * repeats: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
        accelerator.print(f"  num epochs: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        # 3. 创建元数据
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
            "ss_noise_offset_random_strength": args.noise_offset_random_strength,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_c": args.huber_c,
            "ss_fp8_base": bool(args.fp8_base),
            "ss_fp8_base_unet": bool(args.fp8_base_unet),
        }

        self.update_metadata(metadata, args)  # architecture specific metadata

        """4. 处理用户配置"""
        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            """创建数据集元数据 (datasets_metadata)、标签频率 (tag_frequency) 和数据集目录信息 (dataset_dirs_info)。"""
            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # 合并标签频率和数据集目录信息，并将其添加到元数据中。
                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # If a directory is used by multiple datasets, count only once
                    # Since the number of repetitions is originally specified, the number of times a tag appears in the caption does not match the number of times it is used in training.
                    # Therefore, it is not very meaningful to add up the number of times for multiple datasets here.
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # 如果未使用用户配置，则检查数据集组中是否只有一个数据集，并创建相应的元数据。
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug."

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        """5. 添加额外参数和模型名称及哈希"""
        # add extra args
        if args.network_args:
            # 如果指定了网络参数 (args.network_args)，则将其添加到元数据中。
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            # 如果指定了预训练模型名称或路径 (args.pretrained_model_name_or_path)，则计算模型哈希并将其添加到元数据中。
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            # 如果指定了 VAE 名称或路径 (args.vae)，则计算 VAE 哈希并将其添加到元数据中。
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        # 将元数据中的所有值转换为字符串。
        metadata = {k: str(v) for k, v in metadata.items()}

        """6. 创建最小元数据和计算初始步数"""
        # make minimum metadata for filtering
        minimum_metadata = {}
        # 创建最小元数据 (minimum_metadata)，包含过滤所需的最小键值对。
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        """计算初始步数 (initial_step)，根据指定的初始轮数 (args.initial_epoch) 或初始步数 (args.initial_step)，或者从状态文件中加载的步数 (steps_from_state)。
        """
        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
                )
        else:
            # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        # 如果初始步数大于 0，则检查最大训练步数是否大于初始步数。
        if initial_step > 0:
            assert (
                args.max_train_steps > initial_step
            ), f"max_train_steps should be greater than initial step: {args.max_train_steps} vs {initial_step}"

        # 计算开始轮数 (epoch_to_start)，根据初始步数和是否跳过直到初始步数 (args.skip_until_initial_step)。
        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                    )
                logger.info(f"skipping {initial_step} steps")
                initial_step *= args.gradient_accumulation_steps

                # set epoch to start to make initial_step less than len(train_dataloader)
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        
        """7. 获取噪声调度器和初始化跟踪器"""
        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        init_kwargs = {}
        # 初始化跟踪器 (accelerator.init_trackers)，根据指定的 wandb_run_name 或 log_tracker_config 配置跟踪器。
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "network_train" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

        # 初始化损失记录器 (self.loss_recorder)。
        self.loss_recorder = train_util.LossRecorder()
        # 初始化损失记录器 (self.loss_recorder)。
        del train_dataset_group

        pbar.update(1)

        """8. 获取步骤开始回调函数"""
        # callback for step start
        # 检查网络模型是否具有 on_step_start 方法，如果有则获取该方法，否则创建一个空回调函数。
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        """9. 定义保存和删除模型的函数"""
        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            """定义保存模型的函数 (save_model)，创建输出目录，保存模型权重和元数据，并上传到 Hugging Face 仓库（如果指定了 huggingface_repo_id）。"""
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = self.get_sai_model_spec(args)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            """删除模型的函数 (remove_model)，删除指定的旧检查点文件"""
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        """10. 删除不需要的文本编码器"""
        if self.is_text_encoder_not_needed_for_training(args):
            # 如果不需要文本编码器进行训练 (self.is_text_encoder_not_needed_for_training(args))，则删除文本编码器以节省内存。
            logger.info("text_encoder is not needed for training. deleting to save memory.")
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None

        # For --sample_at_first
        #self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)

        """11. 初始化全局步数和打印模型信息"""
        self.global_step = 0
        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            # 初始化全局步数 (self.global_step)，如果指定了初始步数 (initial_step)，则跳过相应的轮数。
            self.global_step = initial_step
            logger.info(f"skipping epoch {epoch_to_start} because initial_step (multiplied) is {initial_step}")
            initial_step -= epoch_to_start * len(train_dataloader)

        # log device and dtype for each model
        logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            params_itr = t_enc.parameters()
            params_itr.__next__()  # skip the first parameter
            params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
            param_3rd = params_itr.__next__()
            logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")
        # 清理加速器设备上的内存 (clean_memory_on_device)。
        clean_memory_on_device(accelerator.device)

        # 设置当前参数到实例上
        self.epoch_to_start = epoch_to_start
        self.num_train_epochs = num_train_epochs
        self.accelerator = accelerator
        self.network = network
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.tokenizers = tokenizers
        self.args = args
        self.train_dataloader = train_dataloader
        self.initial_step = initial_step
        self.current_epoch = current_epoch
        self.metadata = metadata
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_model = save_model
        self.remove_model = remove_model
        self.comfy_pbar = None
        
        progress_bar = tqdm(range(args.max_train_steps - initial_step), smoothing=0, disable=False, desc="steps")
        
        def training_loop(break_at_steps, epoch):
            # steps_done: 用于记录已经完成的训练步数。
            steps_done = 0
            
            # 记录epoch次数+1
            #accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_train_epochs} - steps")

            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            # steps_done: 用于记录已经完成的训练步数。
            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)

            # 如果初始步数 (initial_step) 大于零，则跳过数据加载器中的前几批数据，并将 initial_step 置零。
            skipped_dataloader = None
            if self.initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, self.initial_step)
                self.initial_step = 0
            # 遍历数据加载器中的每一批数据 (batch)。
            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                # 更新当前步数 (current_step.value)。
                current_step.value = self.global_step

                # 使用加速器累积梯度（适用于多 GPU 训练）。
                with accelerator.accumulate(training_model):
                    # 调用 on_step_start 回调函数，进行每一步开始时的操作。
                    on_step_start(text_encoder, unet)

                    # 如果批处理中包含潜在变量 (latents)，则将其移动到目标设备并设置为指定的数据类型。
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        # 否则，使用 VAE 编码图像以生成潜在变量，并进行 NaN 检查和替换操作。
                        with torch.no_grad():
                            # encode latents
                            latents = self.encode_images_to_latents(args, accelerator, vae, batch["images"].to(vae_dtype))
                            latents = latents.to(dtype=weight_dtype)

                            # NaN check
                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.nan_to_num(latents, 0, out=latents)

                    # 对潜在变量进行平移和缩放处理。
                    latents = self.shift_scale_latents(args, latents)

                    # get multiplier for each sample
                    if network_has_multiplier:
                        # 获取每个样本的乘法因子，并设置网络模型的乘法因子。
                        multipliers = batch["network_multipliers"]
                        # if all multipliers are same, use single multiplier
                        if torch.all(multipliers == multipliers[0]):
                            multipliers = multipliers[0].item()
                        else:
                            raise NotImplementedError("multipliers for each sample is not supported yet")
                        # print(f"set multiplier: {multipliers}")
                        accelerator.unwrap_model(network).set_multiplier(multipliers)
                    """初始化条件列表：初始化一个空列表来存储文本编码器条件 (text Encoder Conds) 并从批次中获取预编码的文本编码器输出列表。如果存在预编码输出，则直接赋值给条件列表。"""
                    text_encoder_conds = []
                    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                    if text_encoder_outputs_list is not None:
                        text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs
                    if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
                        """如果条件列表为空或需要训练文本编码器，则启用梯度计算并使用自动混合精度（AMP）上下文环境进行文本嵌入计算。根据配置选择加权嵌入或普通嵌入方法："""
                        with torch.set_grad_enabled(train_text_encoder), accelerator.autocast():
                            # Get the text embedding for conditioning
                            # 加权嵌入：使用特定函数获取加权后的文本嵌入（适用于 SD 模型）。
                            if args.weighted_captions:
                                # SD only
                                encoded_text_encoder_conds = get_weighted_text_embeddings(
                                    tokenizers[0],
                                    text_encoder,
                                    batch["captions"],
                                    accelerator.device,
                                    args.max_token_length // 75 if args.max_token_length else 1,
                                    clip_skip=args.clip_skip,
                                )
                            else:
                                # 普通嵌入：将输入 ID 移动到目标设备，并通过指定策略对令牌进行编码以生成文本嵌入。如果使用全精度浮点类型（FP16），则将生成的嵌入转换为指定的数据类型（weight_dtype）。
                                input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                                encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy,
                                    self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                    input_ids,
                                )
                                # 开启全精度则需要将类型转化成全类型
                                if args.full_fp16:
                                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

                        # 如果没有预编码输出且生成了新嵌入，则直接将新生成的条件赋值给条件列表；
                        # 否则，在存在有效新嵌入的情况下更新已有条件列表中的相应项。
                        # if text_encoder_conds is not cached, use encoded_text_encoder_conds
                        if len(text_encoder_conds) == 0:
                            text_encoder_conds = encoded_text_encoder_conds
                        else:
                            # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                            for i in range(len(encoded_text_encoder_conds)):
                                if encoded_text_encoder_conds[i] is not None:
                                    text_encoder_conds[i] = encoded_text_encoder_conds[i]

                    """噪声预测与目标获取：调用特定方法获取噪声预测、目标值、时间步长、Huber C 和权重因子等信息。这些信息用于后续损失计算过程中的噪声预测模型与真实目标之间的比较。"""
                    # sample noise, call unet, get target
                    noise_pred, target, timesteps, huber_c, weighting = self.get_noise_pred_and_target(
                        args,
                        accelerator,
                        noise_scheduler,
                        latents,
                        batch,
                        text_encoder_conds,
                        unet,
                        network,
                        weight_dtype,
                        train_unet,
                    )

                    """基于噪声预测结果及真实目标值，通过配置文件指定的方法来计算损失函数值。根据具体需求可能还需要应用掩码损失操作或缩放权重等后处理手段以优化最终结果的表现形式。"""
                    loss = train_util.conditional_loss(
                        noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
                    )
                    if weighting is not None:
                        # 应用权重：如果存在权重，则将损失乘以权重。
                        loss = loss * weighting
                    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        # 应用掩码损失：如果启用了掩码损失，则应用掩码损失。
                        loss = apply_masked_loss(loss, batch)
                    # 对损失进行平均。
                    loss = loss.mean([1, 2, 3])

                    # 从批次中获取损失权重。
                    loss_weights = batch["loss_weights"]  # weight for each sample
                    # 将损失乘以损失权重。
                    loss = loss * loss_weights

                    # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                    # 应用不同的损失值函数以加快收敛速度，顺序按照min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation 以此类推
                    loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

                    # 平均损失值
                    loss = loss.mean()  # No need to divide by batch_size since it's an average

                    # 调用 accelerator.backward(loss) 进行反向传播。
                    accelerator.backward(loss)
                    # 如果启用了梯度同步，则调用 self.all_reduce_network 同步梯度。
                    if accelerator.sync_gradients:
                        self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            # 如果启用了梯度裁剪，则进行梯度裁剪。
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    # 调用 optimizer.step() 更新模型参数。
                    optimizer.step()
                    # 调用 lr_scheduler.step() 更新学习率。
                    lr_scheduler.step()
                    # 调用 optimizer.zero_grad(set_to_none=True) 清零梯度。
                    optimizer.zero_grad(set_to_none=True)

                """检查是否启用最大范数正则化
                最大范数正则化通过限制模型参数的范数，防止模型过拟合，提高模型的泛化能力
                通过限制权重向量的大小，可以防止模型过于复杂，从而减少过拟合的风险。
                在每次更新权重后，如果某个权重向量的L2范数超过了预设的最大值 $ c $，就将其缩放回该最大值"""
                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                current_loss = loss.detach().item()
                self.loss_recorder.add(epoch=epoch, step=step, global_step=self.global_step, loss=current_loss)
                avr_loss: float = self.loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions, keys_scaled, mean_norm, maximum_norm
                    )
                    accelerator.log(logs, step=self.global_step)

                if self.global_step >= break_at_steps:
                    break
                steps_done += 1
                self.comfy_pbar.update(1)

            if args.logging_dir is not None:
                logs = {"loss/epoch": self.loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)
  
            return steps_done
        
        

        return training_loop

        # metadata["ss_epoch"] = str(num_train_epochs)
        # metadata["ss_training_finished_at"] = str(time.time())

        # network = accelerator.unwrap_model(network)

        # accelerator.end_training()

        # if (args.save_state or args.save_state_on_train_end):
        #     train_util.save_state_on_train_end(args, accelerator)

        # ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        # save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

        # logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    """创建一个命令行解析器，并添加多个用于控制训练过程的命令行选项。这些选项覆盖了从模型选择到数据集配置等多个方面的需求。"""
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        " / 勾配チェックポイント時にテンソルをCPUにオフロードする（U-NetまたはDiTのみ、サポートされている場合）",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple / Text Encoderの学習率、複数指定可能",
    )
    parser.add_argument(
        "--fp8_base_unet",
        action="store_true",
        help="use fp8 for U-Net (or DiT), Text Encoder is fp16 or bf16"
        " / U-Net（またはDiT）にfp8を使用する。Text Encoderはfp16またはbf16",
    )

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
