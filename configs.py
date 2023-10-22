from typing import Any, List, Optional
from omegaconf import II, MISSING

@dataclass
class ModelConfig():
    pretrained_weights: Optional[str] = field(
        default=None,
        metadata={"help": "pretrained weights for encoder"},
    )

@dataclass
class CorpusConfig():
    path_data: Optional[str] = field(default=None, metadata={"help": "data path"},)
    batch_size: Optional[int] = field(default=None, metadata={"help": "number of examples in a batch"},)
    num_fold: Optional[int] = field(default=None, metadata={"help": "number of folds in cross-validation"},)

@dataclass
class Config:
    corpus: CorpusConfig = CorpusConfig()
    model: ModelConfig = ModelConfig()
    # model: Any = MISSING