import torch
import copy
import math
from typing import Any
import argparse
from .library import flux_models, flux_train_utils, flux_utils, sd3_train_utils, strategy_base, strategy_flux, train_util
from .train_network import NetworkTrainer, clean_memory_on_device, setup_parser

from accelerate import Accelerator


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""该代码文件的主要功能是通过训练带有描述的图像数据来优化模型。代码实现了在训练过程中将模型块在CPU和GPU之间交换，以减少内存使用。此外，代码还使用了融合优化器和梯度钩子来提高梯度处理的效率。该实现基于2kpr的工作，并进行了适应和扩展以满足当前项目的需求。
"""

class FluxNetworkTrainer(NetworkTrainer):
    """封装了训练过程的初始化和训练循环。
    ### 主要功能
    - 继承自 `NetworkTrainer` 类。
        初始化训练参数和数据集。

        设置缓存策略和数据加载器。

        准备加速器和混合精度。

        加载模型和优化器。

        进行训练循环，计算损失并更新模型参数。

        保存训练状态和模型。
"""
    def __init__(self):
        """- 初始化 `FluxNetworkTrainer` 类实例。
- 设置属性 `sample_prompts_te_outputs` 为 None。"""
        super().__init__()
        self.sample_prompts_te_outputs = None

    def assert_extra_args(self, args, train_dataset_group):
        """args (argparse.Namespace): 命令行参数。
        train_dataset_group (DatasetGroup): 训练数据集组。
        * 内部运行逻辑
        - 验证 cache_text_encoder_outputs_to_disk 和 cache_text_encoder_outputs 参数。
        - 验证 cache_text_encoder_outputs 参数。
        - 准备 CLIP-L/T5XXL 训练标志。
        - 验证 max_token_length 参数。
        - 验证 split_mode 和 cpu_offload_checkpointing 参数。
        """
        super().assert_extra_args(args, train_dataset_group)
        # sdxl_train_util.verify_sdxl_training_args(args)

        if args.fp8_base_unet:
            args.fp8_base = True  # if fp8_base_unet is enabled, fp8_base is also enabled for FLUX.1

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        # prepare CLIP-L/T5XXL training flags
        self.train_clip_l = not args.network_train_unet_only
        self.train_t5xxl = False  # default is False even if args.network_train_unet_only is False

        if args.max_token_length is not None:
            logger.warning("max_token_length is not used in Flux training")
        
        assert not args.split_mode or not args.cpu_offload_checkpointing, (
            "split_mode and cpu_offload_checkpointing cannot be used together"
        )

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

    def get_flux_model_name(self, args):
        """功能
            获取 FLUX 模型的名称。
            * 输入
            args (argparse.Namespace): 命令行参数。
            * 内部运行逻辑
            根据 pretrained_model_name_or_path 参数中的关键词判断模型名称。
            * 输出
            返回模型名称（"schnell" 或 "dev"）。"""
        if any(keyword in args.pretrained_model_name_or_path.lower() for keyword in ["schnell", "open", "libre"]):
            return "schnell"
        else:
            return "dev"

    def load_target_model(self, args, weight_dtype, accelerator):
        """- 加载目标 Flux 模型及其相关组件（CLIP-L 和 T5XXL）。
- 处理 FP8 数据类型并根据条件将模型移动到 CPU 或 GPU 上。
* 输入
- args (argparse.Namespace): 命令行参数。
- weight_dtype (torch.dtype): 权重数据类型。
- accelerator (Accelerator): 加速器对象。
* 内部运行逻辑
加载 FLUX 模型、CLIP-L 模型、T5XXL 模型和 VAE 模型。
如果启用了 split_mode，则准备分割模型。
* 输出
返回模型版本、文本编码器、VAE 和 FLUX 模型。"""
        # currently offload to cpu for some models
        name = self.get_flux_model_name(args)

        # if the file is fp8 and we are using fp8_base, we can load it as is (fp8)
        loading_dtype = None if args.fp8_base else weight_dtype

        # if we load to cpu, flux.to(fp8) takes a long time, so we should load to gpu in future
        model = flux_utils.load_flow_model(
            name, args.pretrained_model_name_or_path, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        if args.fp8_base:
            # check dtype of model
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn or model.dtype == torch.float8_e5m2:
                logger.info(f"Loaded {model.dtype} FLUX model")

        if args.split_mode:
            model = self.prepare_split_model(model, args, weight_dtype, accelerator)

        clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        clip_l.eval()

        # if the file is fp8 and we are using fp8_base (not unet), we can load it as is (fp8)
        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None  # as is
        else:
            loading_dtype = weight_dtype

        # loading t5xxl to cpu takes a long time, so we should load to gpu in future
        t5xxl = flux_utils.load_t5xxl(args.t5xxl, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        t5xxl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            # check dtype of model
            if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 T5XXL model")

        ae = flux_utils.load_ae(name, args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)

        return flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model

    def prepare_split_model(self, model, args, weight_dtype, accelerator):
        from accelerate import init_empty_weights

        logger.info("prepare split model")
        with init_empty_weights():
            flux_upper = flux_models.FluxUpper(model.params)
            flux_lower = flux_models.FluxLower(model.params)
        sd = model.state_dict()

        # lower (trainable)
        logger.info("load state dict for lower")
        flux_lower.load_state_dict(sd, strict=False, assign=True)
        flux_lower.to(dtype=weight_dtype)

        # upper (frozen)
        logger.info("load state dict for upper")
        flux_upper.load_state_dict(sd, strict=False, assign=True)

        logger.info("prepare upper model")
        if args.fp8_base:
            if args.fp8_dtype and args.fp8_dtype.lower() == "e5m2":
                target_dtype = torch.float8_e5m2
            else:
                target_dtype = torch.float8_e4m3fn
        else:
            target_dtype =weight_dtype
        flux_upper.to(accelerator.device, dtype=target_dtype)
        flux_upper.eval()

        if args.fp8_base:
            # this is required to run on fp8
            flux_upper = accelerator.prepare(flux_upper)

        flux_upper.to("cpu")

        self.flux_upper = flux_upper
        del model  # we don't need model anymore
        clean_memory_on_device(accelerator.device)

        logger.info("split model prepared")

        return flux_lower

    def get_tokenize_strategy(self, args):
        """获取分词策略。
        * 输入
        - args (argparse.Namespace): 命令行参数。
根据 t5xxl_max_token_length 参数设置分词策略。"""
        name = self.get_flux_model_name(args)

        if args.t5xxl_max_token_length is None:
            if name == "schnell":
                t5xxl_max_token_length = 256
            else:
                t5xxl_max_token_length = 512
        else:
            t5xxl_max_token_length = args.t5xxl_max_token_length

        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
        return strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        """返回文本编码器的分词器列表。"""
        return [tokenize_strategy.clip_l, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        """返回一个用于潜变量缓存的策略对象。"""
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, False)
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        """返回一个用于文本编码的策略对象。"""
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        """后处理网络，检查 T5XXL 是否被训练以及缓存文本编码器输出的相关条件。
        * 输入
            - args (argparse.Namespace): 命令行参数。
            - accelerator (Accelerator): 加速器对象。
            - network (Network): 网络对象。
            - text_encoders (List[torch.nn.Module]): 文本编码器列表。
            - unet (Flux): FLUX 模型。"""
        # check t5xxl is trained or not
        self.train_t5xxl = network.train_t5xxl

        if self.train_t5xxl and args.cache_text_encoder_outputs:
            raise ValueError(
                "T5XXL is trained, so cache_text_encoder_outputs cannot be used / T5XXL学習時はcache_text_encoder_outputsは使用できません"
            )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """获取用于文本编码的模型。
       - 输入
            * args (argparse.Namespace): 命令行参数。
            * accelerator (Accelerator): 加速器对象。
            * text_encoders (List[torch.nn.Module]): 文本编码器列表。
        - 内部运行逻辑
            根据 cache_text_encoder_outputs 参数和训练标志返回相应的文本编码器。
            输出
            返回文本编码器列表。"""
        if args.cache_text_encoder_outputs:
            if self.train_clip_l and not self.train_t5xxl:
                return text_encoders[0:1]  # only CLIP-L is needed for encoding because T5XXL is cached
            else:
                return None  # no text encoders are needed for encoding because both are cached
        else:
            return text_encoders  # both CLIP-L and T5XXL are needed for encoding

    def get_text_encoders_train_flags(self, args, text_encoders):
        """获取文本编码器的训练标志。
        * 输入
            - args (argparse.Namespace): 命令行参数。
            - text_encoders (List[torch.nn.Module]): 文本编码器列表。

        * 内部运行逻辑
            返回 CLIP-L 和 T5XXL 的训练标志。
        输出
            返回训练标志列表。"""
        return [self.train_clip_l, self.train_t5xxl]

    def get_text_encoder_outputs_caching_strategy(self, args):
        """* 功能
        - 获取文本编码器输出的缓存策略。
        * 输入
            - args (argparse.Namespace): 命令行参数。
        * 内部运行逻辑
            - 根据 cache_text_encoder_outputs 参数和训练标志设置文本编码器输出的缓存策略。
        * 输出
            - 返回文本编码器输出的缓存策略对象。"""
        if args.cache_text_encoder_outputs:
            # if the text encoders is trained, we need tokenization, so is_partial is True
            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                None,
                False,
                is_partial=self.train_clip_l or self.train_t5xxl,
                apply_t5_attn_mask=args.apply_t5_attn_mask,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        """* 功能
        - 如果需要，缓存文本编码器的输出。
        * 输入
            - args (argparse.Namespace): 命令行参数。
            - accelerator (Accelerator): 加速器对象。
            - unet (Flux): FLUX 模型。
            - vae (VAE): VAE 模型。
            - text_encoders (List[torch.nn.Module]): 文本编码器列表。
            - dataset (DatasetGroup): 数据集组。
            - weight_dtype (torch.dtype): 权重数据类型。
        * 内部运行逻辑
        如果启用了 cache_text_encoder_outputs，则缓存文本编码器的输出。
        如果需要，缓存样本提示的嵌入。
        输出
        无。"""
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # reduce memory consumption
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            logger.info("move text encoders to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)  # always not fp8
            text_encoders[1].to(accelerator.device)

            if text_encoders[1].dtype == torch.float8_e4m3fn:
                # if we load fp8 weights, the model is already fp8, so we use it as is
                self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype)
            else:
                # otherwise, we need to convert it to target dtype
                text_encoders[1].to(weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator.is_main_process)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = []
                for line in args.sample_prompts:
                    line = line.strip()
                    if len(line) > 0 and line[0] != "#":
                        prompts.append(line)
                
                # preprocess prompts
                for i in range(len(prompts)):
                    prompt_dict = prompts[i]
                    if isinstance(prompt_dict, str):
                        from .library.train_util import line_to_prompt_dict

                        prompt_dict = line_to_prompt_dict(prompt_dict)
                        prompts[i] = prompt_dict
                    assert isinstance(prompt_dict, dict)

                    # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
                    prompt_dict["enum"] = i
                    prompt_dict.pop("subset", None)

                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks, args.apply_t5_attn_mask
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move back to cpu
            if not self.is_train_text_encoder(args):
                logger.info("move CLIP-L back to cpu")
                text_encoders[0].to("cpu")
            logger.info("move t5XXL back to cpu")
            text_encoders[1].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoder
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device)

    def sample_images(self, accelerator, args, epoch, global_step, flux, ae, text_encoder, sample_prompts_te_outputs, validation_settings):
        """* 功能
        生成样本图像。
        * 输入
        - accelerator (Accelerator): 加速器对象。
        - args (argparse.Namespace): 命令行参数。
        - epoch (int): 当前周期。
        - global_step (int): 当前全局步数。
        - flux (Flux): FLUX 模型。
        - ae (VAE): VAE 模型。
        - text_encoder (List[torch.nn.Module]): 文本编码器列表。
        - sample_prompts_te_outputs (Dict): 样本提示的文本编码器输出。
        - validation_settings (Dict): 验证设置。
        * 内部运行逻辑
        如果启用了 split_mode，则使用分割模型生成图像。
        否则，直接使用 FLUX 模型生成图像。

        * 输出
        返回生成的图像张量。"""
        text_encoders = text_encoder  # for compatibility
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        if not args.split_mode:
            image_tensors = flux_train_utils.sample_images(
            accelerator, args, epoch, global_step, flux, ae, text_encoders, sample_prompts_te_outputs, validation_settings)
            clean_memory_on_device(accelerator.device)
            return image_tensors
        
        class FluxUpperLowerWrapper(torch.nn.Module):
            """FluxUpperLowerWrapper 类是一个内部类，用于在分割模式（split mode）下包裹 FluxUpper 和 FluxLower 模型。这个类的主要目的是在前向传播过程中，动态地将模型的上层和下层模型在 CPU 和 GPU 之间进行切换，以减少内存占用。
            
            """
            def __init__(self, flux_upper: flux_models.FluxUpper, flux_lower: flux_models.FluxLower, device: torch.device):
                """- 初始化 `FluxUpperLowerWrapper` 类实例。
                    - 参数：
                    - `flux_upper`: Flux 上层模型。
                    - `flux_lower`: Flux 下层模型。
                    - `device`: 目标设备（通常为 GPU）。"""
                super().__init__()
                self.flux_upper = flux_upper
                self.flux_lower = flux_lower
                self.target_device = device

            def forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None, txt_attention_mask=None):
                """ 定义前向传播方法。
                - 参数：
                - `img`: 输入图像特征。
                - `img_ids`: 图像 ID。
                - `txt`: 文本特征。
                - `txt_ids`: 文本 ID。
                - `timesteps`: 时间步长。
                - `y`: 其他输入参数。
                - `guidance`: 引导参数（可选）。
                - `txt_attention_mask`: 文本注意力掩码（可选）。

                ### 内部逻辑
                1. **切换设备**：
                ```python
                self.flux_lower.to("cpu")
                clean_memory_on_device(self.target_device)
                self.flux_upper.to(self.target_device)
                ```
                在前向传播开始时，将下层模型移动到 CPU 并清理目标设备上的内存，然后将上层模型移动到目标设备。
"""
                self.flux_lower.to("cpu")
                clean_memory_on_device(self.target_device)
                self.flux_upper.to(self.target_device)
                img, txt, vec, pe = self.flux_upper(img, img_ids, txt, txt_ids, timesteps, y, guidance, txt_attention_mask)
                self.flux_upper.to("cpu")
                clean_memory_on_device(self.target_device)
                self.flux_lower.to(self.target_device)
                return self.flux_lower(img, txt, vec, pe, txt_attention_mask)

        wrapper = FluxUpperLowerWrapper(self.flux_upper, flux, accelerator.device)
        clean_memory_on_device(accelerator.device)
        image_tensors = flux_train_utils.sample_images(
            accelerator, args, epoch, global_step, wrapper, ae, text_encoders, sample_prompts_te_outputs, validation_settings
        )
        clean_memory_on_device(accelerator.device)
        return image_tensors

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        """* 功能
        - 获取噪声调度器。
        * 输入
        - args (argparse.Namespace): 命令行参数。
        - device (torch.device): 设备对象。
        * 内部运行逻辑
        - 创建并返回噪声调度器。
        * 输出
        - 返回噪声调度器对象。"""
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, accelerator, vae, images):
        """功能
        - 将图像编码为潜在变量。
        * 输入
            - args (argparse.Namespace): 命令行参数。
            - accelerator (Accelerator): 加速器对象。
            - vae (VAE): VAE 模型。
            - images (torch.Tensor): 输入图像张量。
        * 内部运行逻辑
            - 使用 VAE 模型将图像编码为潜在变量。
        * 输出
            - 返回潜在变量张量。"""
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        """功能
缩放和偏移潜在变量。
输入
args (argparse.Namespace): 命令行参数。
latents (torch.Tensor): 潜在变量张量。
内部运行逻辑
直接返回输入的潜在变量张量。
输出
返回潜在变量张量。"""
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
    ):
        """* 功能
            获取噪声预测和目标。
            * 输入
                args (argparse.Namespace): 命令行参数。
                accelerator (Accelerator): 加速器对象。
                noise_scheduler (NoiseScheduler): 噪声调度器对象。
                latents (torch.Tensor): 潜在变量张量。
                batch (Dict): 批次数据。
                text_encoder_conds (List[torch.Tensor]): 文本编码器条件张量。
                unet (Flux): FLUX 模型。
                network (Network): 网络对象。
                weight_dtype (torch.dtype): 权重数据类型。
                train_unet (bool): 是否训练 UNet。

            * 内部运行逻辑
                生成噪声并添加到潜在变量中。
                获取噪声模型输入和时间步。
                打包潜在变量并获取图像 ID。
                获取指导向量。
                预测噪声残差。
                解包潜在变量。
                应用模型预测类型。
                获取目标噪声。
            - 输出
            * 返回噪声预测、目标噪声、时间步、权重和加权方案。"""
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        # pack latents and get img_ids
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # get guidance
        # ensure guidance_scale in args is float
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t.dtype.is_floating_point:
                    t.requires_grad_(True)
            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        # Predict the noise residual
        l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        if not args.split_mode:
            # normal forward
            with accelerator.autocast():
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                model_pred = unet(
                    img=packed_noisy_model_input,
                    img_ids=img_ids,
                    txt=t5_out,
                    txt_ids=txt_ids,
                    y=l_pooled,
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec,
                    txt_attention_mask=t5_attn_mask,
                )
        else:
            # split forward to reduce memory usage
            assert network.train_blocks == "single", "train_blocks must be single for split mode"
            with accelerator.autocast():
                # move flux lower to cpu, and then move flux upper to gpu
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)
                self.flux_upper.to(accelerator.device)

                # upper model does not require grad
                with torch.no_grad():
                    intermediate_img, intermediate_txt, vec, pe = self.flux_upper(
                        img=packed_noisy_model_input,
                        img_ids=img_ids,
                        txt=t5_out,
                        txt_ids=txt_ids,
                        y=l_pooled,
                        timesteps=timesteps / 1000,
                        guidance=guidance_vec,
                        txt_attention_mask=t5_attn_mask,
                    )

                # move flux upper back to cpu, and then move flux lower to gpu
                self.flux_upper.to("cpu")
                clean_memory_on_device(accelerator.device)
                unet.to(accelerator.device)

                # lower model requires grad
                intermediate_img.requires_grad_(True)
                intermediate_txt.requires_grad_(True)
                vec.requires_grad_(True)
                pe.requires_grad_(True)
                model_pred = unet(img=intermediate_img, txt=intermediate_txt, vec=vec, pe=pe, txt_attention_mask=t5_attn_mask)

        # unpack latents
        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        # flow matching loss: this is different from SD3
        target = noise - latents

        return model_pred, target, timesteps, None, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        """后处理loss值"""
        return loss

    def get_sai_model_spec(self, args):
        """功能
获取 SAI 模型规范。

输入
args (argparse.Namespace): 命令行参数。

内部运行逻辑
返回 SAI 模型规范。"""
        return train_util.get_sai_model_spec(None, args, False, True, False, flux="dev")

    def update_metadata(self, metadata, args):
        """* 功能
            更新元数据。
        * 输入
            metadata (Dict): 元数据字典。
            args (argparse.Namespace): 命令行参数。
        * 内部运行逻辑
            更新元数据字典中的相关参数。
        * 输出
        无。"""
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        """* 功能
            判断是否不需要文本编码器进行训练。
        * 输入
            args (argparse.Namespace): 命令行参数。
        * 内部运行逻辑
            根据 cache_text_encoder_outputs 参数和训练标志判断是否不需要文本编码器进行训练。
        * 输出
            返回布尔值。"""
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        """* 功能
            - 准备文本编码器的梯度检查点工作区。

        * 输入
            - index (int): 文本编码器索引。
            - text_encoder (torch.nn.Module): 文本编码器对象。

        * 内部运行逻辑
            根据文本编码器索引准备相应的梯度检查点工作区。

        * 输出
            无。"""
        if index == 0:  # CLIP-L
            return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)
        else:  # T5XXL
            text_encoder.encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        """* 功能
            准备文本编码器的 FP8 训练。
            * 输入
                index (int): 文本编码器索引。
                text_encoder (torch.nn.Module): 文本编码器对象。
                te_weight_dtype (torch.dtype): 文本编码器权重数据类型。
                weight_dtype (torch.dtype): 权重数据类型。
            * 内部运行逻辑
                根据文本编码器索引准备相应的 FP8 训练。
            * 输出
            无。

        """
        if index == 0:  # CLIP-L
            logger.info(f"prepare CLIP-L for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}")
            text_encoder.to(te_weight_dtype)  # fp8
            text_encoder.text_model.embeddings.to(dtype=weight_dtype)
        else:  # T5XXL
            def prepare_fp8(text_encoder, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        hidden_gelu = module.act(module.wi_0(hidden_states))
                        hidden_linear = module.wi_1(hidden_states)
                        hidden_states = hidden_gelu * hidden_linear
                        hidden_states = module.dropout(hidden_states)

                        hidden_states = module.wo(hidden_states)
                        return hidden_states

                    return forward

                for module in text_encoder.modules():
                    if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = forward_hook(module)

            if flux_utils.get_t5xxl_actual_dtype(text_encoder) == torch.float8_e4m3fn and text_encoder.dtype == weight_dtype:
                logger.info(f"T5XXL already prepared for fp8")
            else:
                logger.info(f"prepare T5XXL for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}, add hooks")
                text_encoder.to(te_weight_dtype)  # fp8
                prepare_fp8(text_encoder, weight_dtype)


def setup_parser() -> argparse.ArgumentParser:
    """* 功能
        设置命令行参数解析器。
    * 输入
        无。
    * 内部运行逻辑
        创建一个 argparse.ArgumentParser 对象。
    * 输出
        返回配置好的 argparse.ArgumentParser 对象。"""
    parser = setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument(
        "--split_mode",
        action="store_true",
        help="[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = FluxNetworkTrainer()
    trainer.train(args)