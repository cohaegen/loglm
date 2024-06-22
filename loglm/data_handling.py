import tensorflow as tf
import keras


RECORD_SEPARATOR = '<RS>'


def window_dataset(ds: tf.data.Dataset, context: int) -> tf.data.Dataset:
    """
    Create a series of sliding time windows from a dataset that is a serial, 1-d
    sequence of tokens
    """
    ts_ds = ds.window(context, shift=1, drop_remainder=True)\
              .flat_map(lambda x: x.batch(context))
    return ts_ds


def create_value_target_dataset(ds: tf.data.Dataset,
                                context: int,
                                batch_size: int,
                                shuffle: bool=False,
                                shuffle_buffer_size: int=0) -> tf.data.Dataset:
    """
    Create a dataset that has values and targets for training
    ds is a dataset with a serial sequence of tokens
    Values and targets are sliding time windows from ds
    Values are the x values for the model to use as inputs for prediction
    Targets are the y values for the model to predict. They are the value
    tokens shifted right in time by one (so value tokens are used to predict the next 
    tokens in time which are the targets)
    """
    values = window_dataset(ds, context)
    targets = window_dataset(ds.skip(1), context)
    train_ds = tf.data.Dataset.zip(values, targets)
    if shuffle:
        assert shuffle_buffer_size > 0
        train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds


def read_text_file(filenames: str,
                   encoder: keras.layers.StringLookup,
                   adapt_encoder: bool=False,
                   record_separator: str=RECORD_SEPARATOR):
    file_ds = tf.data.TextLineDataset(filenames)
    # Split into bytes
    file_ds = file_ds.map(lambda x: tf.concat([tf.strings.bytes_split(x), tf.constant([record_separator])], axis=0))
    file_ds = file_ds.unbatch()
    if adapt_encoder:
        encoder.adapt(file_ds.batch(1_000).take(1_000))
    file_ds = file_ds.map(encoder)
    return file_ds