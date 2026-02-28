import importlib
import os
import torch
from torch import nn
from enum import Enum
from typing import Optional, Type, List, Dict, TYPE_CHECKING

from aegaeon.logger import init_logger

if TYPE_CHECKING:
    from aegaeon.config import ModelConfig, ParallelConfig
    from aegaeon.loader import QuickLoader


logger = init_logger(__name__)


class ModelType(Enum):

    qwen_7b = 0
    qwen_7b_chat = 1
    qwen2_5_7b = 2
    llama2_7b = 3
    yi1_5_6b_chat = 4
    qwen1_5_14b_chat = 5
    llama2_13b_chat = 6
    qwen_14b_chat = 7
    internlm2_5_7b_chat = 8
    llava1_5_13b = 9
    qwen1_5_moe_a2_7b_chat = 10
    yi1_5_9b_chat = 11
    qwen2_1_5b = 12
    qwen2_5_72b = 13

    synth_model_00 = 10000
    synth_model_01 = 10001
    synth_model_02 = 10002
    synth_model_03 = 10003
    synth_model_04 = 10004
    synth_model_05 = 10005
    synth_model_06 = 10006
    synth_model_07 = 10007
    synth_model_08 = 10008
    synth_model_09 = 10009
    synth_model_10 = 10010
    synth_model_11 = 10011
    synth_model_12 = 10012
    synth_model_13 = 10013
    synth_model_14 = 10014
    synth_model_15 = 10015
    synth_model_16 = 10016
    synth_model_17 = 10017
    synth_model_18 = 10018
    synth_model_19 = 10019
    synth_model_20 = 10020
    synth_model_21 = 10021
    synth_model_22 = 10022
    synth_model_23 = 10023
    synth_model_24 = 10024
    synth_model_25 = 10025
    synth_model_26 = 10026
    synth_model_27 = 10027
    synth_model_28 = 10028
    synth_model_29 = 10029
    synth_model_30 = 10030
    synth_model_31 = 10031
    synth_model_32 = 10032
    synth_model_33 = 10033
    synth_model_34 = 10034
    synth_model_35 = 10035
    synth_model_36 = 10036
    synth_model_37 = 10037
    synth_model_38 = 10038
    synth_model_39 = 10039
    synth_model_40 = 10040
    synth_model_41 = 10041
    synth_model_42 = 10042
    synth_model_43 = 10043
    synth_model_44 = 10044
    synth_model_45 = 10045
    synth_model_46 = 10046
    synth_model_47 = 10047
    synth_model_48 = 10048
    synth_model_49 = 10049
    synth_model_50 = 10050
    synth_model_51 = 10051
    synth_model_52 = 10052
    synth_model_53 = 10053
    synth_model_54 = 10054
    synth_model_55 = 10055
    synth_model_56 = 10056
    synth_model_57 = 10057
    synth_model_58 = 10058
    synth_model_59 = 10059
    synth_model_60 = 10060
    synth_model_61 = 10061
    synth_model_62 = 10062
    synth_model_63 = 10063
    synth_model_64 = 10064
    synth_model_65 = 10065
    synth_model_66 = 10066
    synth_model_67 = 10067
    synth_model_68 = 10068
    synth_model_69 = 10069
    synth_model_70 = 10070
    synth_model_71 = 10071
    synth_model_72 = 10072
    synth_model_73 = 10073
    synth_model_74 = 10074
    synth_model_75 = 10075
    synth_model_76 = 10076
    synth_model_77 = 10077
    synth_model_78 = 10078
    synth_model_79 = 10079

    def __str__(self):
        return self.name

    @staticmethod
    def from_int(i: int) -> Optional["ModelType"]:
        return next((m for m in ModelType if m.value == i), None)

    @staticmethod
    def from_str(s: str, default: "ModelType" = None) -> Optional["ModelType"]:
        if s.startswith("synth_model"):
            return ModelType.from_int(10000 + int(s[12:]))
        if s in ("qwen_7b", "qwen-7b"):
            return ModelType.qwen_7b
        if s in ("qwen_7b_chat", "qwen-7b-chat"):
            return ModelType.qwen_7b_chat
        if s in ("qwen2_5_7b", "qwen2.5-7b"):
            return ModelType.qwen2_5_7b
        if s in ("llama2_7b_chat", "llama2-7b-chat"):
            return ModelType.llama2_7b
        if s in ("yi1_5_6b_chat", "yi1.5-6b-chat"):
            return ModelType.yi1_5_6b_chat
        if s in ("qwen1_5_14b_chat", "qwen1.5-14b-chat"):
            return ModelType.qwen1_5_14b_chat
        if s in ("llama2_13b_chat", "llama2-13b-chat"):
            return ModelType.llama2_13b_chat
        if s in ("qwen_14b_chat", "qwen-14b-chat"):
            return ModelType.qwen_14b_chat
        if s in ("internlm2_5_7b_chat", "internlm2.5-7b-chat"):
            return ModelType.internlm2_5_7b_chat
        if s in ("llava1_5_13b", "llava1.5-13b"):
            return ModelType.llava1_5_13b
        if s in ("qwen1_5_moe_a2_7b_chat", "qwen1.5-moe-a2.7b-chat"):
            return ModelType.qwen1_5_moe_a2_7b_chat
        if s in ("yi1_5_9b_chat", "yi1.5-9b-chat"):
            return ModelType.yi1_5_9b_chat
        if s in ("qwen2_1_5b", "qwen2-1.5b"):
            return ModelType.qwen2_1_5b
        if s in ("qwen2_5_72b", "qwen2.5-72b"):
            return ModelType.qwen2_5_72b

        return default

    def alias(self) -> "ModelType":
        if self.value < 10000:
            return self
        i = (self.value - 10000) % 10
        return [
            ModelType.qwen2_5_7b,
            ModelType.qwen2_5_7b,
            ModelType.qwen2_5_7b,
            ModelType.qwen2_5_7b,
            ModelType.qwen2_5_7b,
            ModelType.qwen2_5_7b,
            ModelType.yi1_5_9b_chat,
            ModelType.yi1_5_9b_chat,
            ModelType.llama2_13b_chat,
            ModelType.llama2_13b_chat,
        ][i]

    def path(self) -> str:
        if self.value >= 10000:
            return self.alias().path()
        match self:
            case ModelType.qwen_7b:
                return "/root/models/Qwen-7B/"
            case ModelType.qwen_7b_chat:
                return "/root/models/Qwen-7B-Chat/"
            case ModelType.qwen2_5_7b:
                return "/root/models/Qwen2.5-7B-Instruct/"
            case ModelType.llama2_7b:
                return "/root/models/Llama-2-7b-hf/"
            case ModelType.internlm2_5_7b_chat:
                return "/root/models/internlm2_5-7b-chat/"
            case ModelType.yi1_5_6b_chat:
                return "/root/models/Yi-1.5-6B-Chat/"
            case ModelType.qwen1_5_14b_chat:
                return "/root/models/Qwen1.5-14B-Chat/"
            case ModelType.llama2_13b_chat:
                return "/root/models/Llama-2-13b-chat-ms/"
            case ModelType.qwen_14b_chat:
                return "/root/models/Qwen-14B-Chat/"
            case ModelType.llava1_5_13b:
                return "/root/models/llava-1.5-13b-hf/"
            case ModelType.qwen1_5_moe_a2_7b_chat:
                return "/root/models/Qwen1.5-MoE-A2.7B-Chat/"
            case ModelType.yi1_5_9b_chat:
                return "/root/models/Yi-1.5-9B-Chat/"
            case ModelType.qwen2_1_5b:
                return "/root/models/Qwen2-1.5B/"
            case ModelType.qwen2_5_72b:
                return "/root/models/Qwen2.5-72B-Instruct/"
            case _:
                raise NotImplementedError(f"{self}")

    def n_params(self) -> int:
        if self.value >= 10000:
            return self.alias().n_params()
        match self:
            case (
                ModelType.qwen_7b
                | ModelType.qwen_7b_chat
                | ModelType.internlm2_5_7b_chat
                | ModelType.qwen2_5_7b
            ):
                return int(7.25 * (1024**3))
            case ModelType.yi1_5_6b_chat:
                return int(5.75 * (1024**3))
            case ModelType.llama2_7b:
                return int(6.3 * (1024**3))
            case ModelType.yi1_5_9b_chat:
                return int(8.25 * (1024**3))
            case ModelType.llama2_13b_chat:
                return int(12.2 * (1024**3))
            case ModelType.llava1_5_13b:
                return int(12.5 * (1024**3))
            case ModelType.qwen1_5_14b_chat | ModelType.qwen_14b_chat:
                return int(13.25 * (1024**3))
            case ModelType.qwen1_5_moe_a2_7b_chat:
                return int(13.4 * (1024**3))
            case ModelType.qwen2_1_5b:
                return int(1.67 * (1024**3))
            case ModelType.qwen2_5_72b:
                return int(72 * (1024**3))
            case _:
                raise NotImplementedError(f"{self}")


# Architecture -> (module, class).
_GENERATION_MODELS = {
    "AquilaModel": ("llama", "LlamaForCausalLM"),
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),  # AquilaChat2
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),  # baichuan-7b
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),  # baichuan-13b
    # "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    # "CohereForCausalLM": ("commandr", "CohereForCausalLM"),
    # "DbrxForCausalLM": ("dbrx", "DbrxForCausalLM"),
    # "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    # "DeepseekForCausalLM": ("deepseek", "DeepseekForCausalLM"),
    # "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    # "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    # "Gemma2ForCausalLM": ("gemma2", "Gemma2ForCausalLM"),
    # "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    # "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    # "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    # "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "InternLMForCausalLM": ("llama", "LlamaForCausalLM"),
    "InternLM2ForCausalLM": ("internlm2", "InternLM2ForCausalLM"),
    # "JAISLMHeadModel": ("jais", "JAISLMHeadModel"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    "LlavaNextForConditionalGeneration": (
        "llava_next",
        "LlavaNextForConditionalGeneration",
    ),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    # "QuantMixtralForCausalLM": ("mixtral_quant", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    # "MptForCausalLM": ("mpt", "MPTForCausalLM"),
    # "MPTForCausalLM": ("mpt", "MPTForCausalLM"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMForCausalLM"),
    # "OlmoForCausalLM": ("olmo", "OlmoForCausalLM"),
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    # "OrionForCausalLM": ("orion", "OrionForCausalLM"),
    # "PhiForCausalLM": ("phi", "PhiForCausalLM"),
    # "Phi3ForCausalLM": ("llama", "LlamaForCausalLM"),
    # "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
    "QWenLMHeadModel": ("qwen", "QWenLMHeadModel"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    # "RWForCausalLM": ("falcon", "FalconForCausalLM"),
    # "StableLMEpochForCausalLM": ("stablelm", "StablelmForCausalLM"),
    # "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    # "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    # "ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
    # "XverseForCausalLM": ("xverse", "XverseForCausalLM"),
    # "Phi3SmallForCausalLM": ("phi3_small", "Phi3SmallForCausalLM"),
    # "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
    # "JambaForCausalLM": ("jamba", "JambaForCausalLM")
}

_EMBEDDING_MODELS = {}

_MODELS = {**_GENERATION_MODELS, **_EMBEDDING_MODELS}

# Architecture -> type.
# out of tree models
_OOT_MODELS: Dict[str, Type[nn.Module]] = {}


class ModelRegistry:
    """
    Imports directly from vllm-adapted models.
    """

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None

        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(f"vllm.model_executor.models.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())

    @staticmethod
    def register_model(model_arch: str, model_cls: Type[nn.Module]):
        if model_arch in _MODELS:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.",
                model_arch,
                model_cls.__name__,
            )
        global _OOT_MODELS
        _OOT_MODELS[model_arch] = model_cls

    @staticmethod
    def is_embedding_model(model_arch: str) -> bool:
        return model_arch in _EMBEDDING_MODELS


@torch.inference_mode()
def get_model(
    *,
    model_config: "ModelConfig",
    parallel_config: "ParallelConfig",
    device: torch.device,
    quick_loader: Optional["QuickLoader"] = None,
    **kwargs,
) -> nn.Module:
    from aegaeon.loader import get_model_loader

    if parallel_config.world_size > 1:
        raise NotImplementedError("tp>1")

    extra_model_kwargs = {**kwargs}
    # XXX: ugly fix..
    if model_config.model.alias() in (ModelType.llava1_5_13b,):
        from vllm.config import MultiModalConfig

        extra_model_kwargs["multimodal_config"] = MultiModalConfig()
    if model_config.model.alias() in (ModelType.qwen2_5_7b,):
        from vllm.config import CacheConfig
        from aegaeon.config import BLOCK_SIZE

        extra_model_kwargs["cache_config"] = CacheConfig(
            # None of these fields actually matters
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0,
            swap_space=0,
            cache_dtype="auto",
        )

    loader = get_model_loader(quick_loader)

    return loader.load_model(
        model_config=model_config, device=device, **extra_model_kwargs
    )
