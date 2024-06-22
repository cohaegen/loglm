from dataclasses import dataclass
import keras
from .transformers import TransformerDecoder, TokenAndPositionEmbedding


@dataclass
class XFormerADConfig:
    context_size: int
    num_heads: int
    head_size: int
    num_layers: int
    dropout: float
    vocabulary_size: int


class XformerAD(keras.Sequential):
    def __init__(self, config: XFormerADConfig):
        super().__init__()
        self.add(keras.layers.Input(config.context_size,))
        self.add(TokenAndPositionEmbedding(config.vocabulary_size, config.num_heads*config.head_size))
        for _ in range(config.num_layers):
            self.add(TransformerDecoder(config.num_heads, config.head_size, config.dropout))
        self.add(keras.layers.LayerNormalization())
        self.add(keras.layers.Dense(config.vocabulary_size))