from pathlib import Path
from aenum import Enum, NoAlias, extend_enum
from typing_extensions import Optional

SIMDATA = Path(__file__).parent.absolute().joinpath("simdata")

class DeviceType(Enum):
    a10 = "a10"
    l20 = "l20"
    a100 = "a100"
    h800 = "h800"
    h100 = "h100"

    def __str__(self):
        return self.value
    
    def mem_capacity(self):
        """GPU memory capacity in GB"""
        match self:
            case DeviceType.a10: return 24
            case DeviceType.l20: return 48
            case DeviceType.a100: return 80
            case DeviceType.h800: return 80
            case DeviceType.h100: return 80
            case _: raise NotImplementedError(f"mem_capacity not implemented for {self}")

class ModelType(Enum):

    _settings_ = NoAlias

    yi_9b = "yi-9b"
    llama2_13b = "llama2-13b"
    qwen2_5_7b = "qwen2.5-7b"


    @staticmethod
    def extend_alias(canonical: str, alias: str) -> 'ModelType':
        if ModelType.from_str(alias) is None:
            raise ValueError(f'{alias} is not a valid alias for {canonical}')
        extend_enum(ModelType, canonical, alias)
        return ModelType[canonical]
            
    @staticmethod
    def from_str(s: str) -> Optional['ModelType']:
        try:
            return ModelType[s]
        except:
            for e in ModelType:
                if e.value == s:
                    return e
            return None
    
    def as_str(self) -> str:
        return self.value
    
    def mem_size(self) -> float:
        """Model memory size in GB"""
        match self:
            case ModelType.qwen2_5_7b: return 14
            case ModelType.yi_9b: return 18
            case ModelType.llama2_13b: return 26
            case _: 
                if (alias := ModelType.from_str(self.value)) is not None:
                    return alias.mem_size()
                raise NotImplementedError(f"mem_size not implemented for {self}")
    
    def kvcache_size(self) -> float:
        """KV cache per token size in GB"""
        match self:
            case ModelType.qwen2_5_7b: return 4 * 28 * 3584 / (1024**3)
            case ModelType.yi_9b: return 4 * 48 * 4096 / (1024**3)
            case ModelType.llama2_13b: return 4 * 40 * 5120 / (1024**3)
            case _: 
                if (alias := ModelType.from_str(self.value)) is not None:
                    return alias.kvcache_size()
                raise NotImplementedError(f"kvcache_size not implemented for {self}")

    def max_seq_len(self) -> int:
        match self:
            case ModelType.qwen2_5_7b: return 8192
            case ModelType.yi_9b: return 4096
            case ModelType.llama2_13b: return 4096
            case _: 
                if (alias := ModelType.from_str(self.value)) is not None:
                    return alias.max_seq_len()
                raise NotImplementedError(f"max_seq_len not implemented for {self}")
    
    def max_num_tokens(self, device: DeviceType = DeviceType.a10) -> int:
        KVCACHE_PERCENTAGE = 0.8
        return int((device.mem_capacity() - self.mem_size()) * KVCACHE_PERCENTAGE / self.kvcache_size())
