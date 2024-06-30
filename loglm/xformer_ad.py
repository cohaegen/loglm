from dataclasses import dataclass
import tensorflow as tf
import keras
from .transformers import TransformerDecoder, TokenAndPositionEmbedding, InvertibleEmbedding


@dataclass
class XFormerADConfig:
    context_size: int
    num_heads: int
    head_size: int
    num_layers: int
    dropout: float
    vocabulary_size: int


class XformerAD(keras.Model):
    def __init__(self, config: XFormerADConfig) -> None:
        # Input is a tensor of integers that represent token IDs. The shape is the length of the context window we're using
        inp = keras.layers.Input((config.context_size,))
        # Create token and position embeddings and add them.
        # The token embedding is invertible because we will re-use it at the end of the model.
        tok_embedding = InvertibleEmbedding(config.vocabulary_size,
                                            config.num_heads*config.head_size)
        # Just use static values for the positions
        positions = tf.range(config.context_size)
        pos_embedding = keras.layers.Embedding(config.context_size,
                                               config.num_heads*config.head_size)
        layer = tok_embedding(inp) + pos_embedding(positions)
        # Add transformer decoder layers. They comprise an attention block and Dense layer. They use a causal mask.
        for _ in range(config.num_layers):
            layer = TransformerDecoder(config.num_heads, config.head_size, config.dropout)(layer)
        # Layer norm before finishing
        layer = keras.layers.LayerNormalization()(layer)
        # This is the final step: instead of a Dense layer, we're re-using the weights from the token embedding.
        layer = tok_embedding(layer, invert=True)
        # Initialize as a keras.Model
        super().__init__(inputs=inp, outputs=layer)