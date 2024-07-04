import argparse
import keras
import loglm


TIMESERIES_CONTEXT = 8
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1_000
NUM_HEADS = 4
HEAD_SIZE = 16
NUM_LAYERS = 4
DROPOUT = 0.2
LEARNING_RATE = 3e-4


def main(args: argparse.Namespace=None):
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('input_file_pattern', help='Input files')
        parser.add_argument('model_file')
        args = parser.parse_args()

    encoder = keras.layers.StringLookup()
    file_ds = loglm.read_text_file(args.input_file_pattern, encoder, adapt_encoder=True)
    train_ds = loglm.create_value_target_dataset(file_ds,
                                                 TIMESERIES_CONTEXT,
                                                 BATCH_SIZE,
                                                 shuffle=True,
                                                 shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)

    model = loglm.XformerAD(loglm.XFormerADConfig(context_size=TIMESERIES_CONTEXT,
                                                  num_heads=NUM_HEADS,
                                                  head_size=HEAD_SIZE,
                                                  num_layers=NUM_LAYERS,
                                                  dropout=DROPOUT,
                                                  vocabulary_size=encoder.vocabulary_size()))

    optimizer = keras.optimizers.AdamW(LEARNING_RATE)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer, loss)
    model.summary()

    model.fit(train_ds.take(10_000))


if __name__ == '__main__':
    main()