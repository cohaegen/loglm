import loglm
import keras
import tensorflow as tf
import IPython

TIMESERIES_CONTEXT = 32
NUM_HEADS = 4
HEAD_SIZE = 32
NUM_LAYERS = 6
DROPOUT = 0.2
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
# SHUFFLE_BUFFER_SIZE = 1_000_000
SHUFFLE_BUFFER_SIZE = 10_000


encoder = keras.layers.StringLookup()
file_ds = loglm.read_text_file('../projects/Monday-WorkingHours.csv', encoder, adapt_encoder=True)
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

model.fit(train_ds)
model.save('iscx.keras')

test_file_ds = loglm.read_text_file('./Monday-WorkingHours_test.csv', encoder, adapt_encoder=False)
test_ds = loglm.create_value_target_dataset(test_file_ds,
                                            TIMESERIES_CONTEXT,
                                            BATCH_SIZE,
                                            shuffle=False)
def anomaly_scores(model: keras.Model, vt_ds: tf.data.Dataset, decoder: keras.layers.StringLookup):
    decode = lambda x: b''.join(decoder(x).numpy()).decode()
    for v, t in vt_ds:
        pred = model.predict(v, verbose=0)
        anomaly_scores = pred[range(len(v)), -1, t[:,-1]]
        decoded_target = decode(t[:,-1])
        print(f'{decoded_target.ljust(60, " ")} mean_score: {anomaly_scores.mean():0.4f} min_score: {anomaly_scores.min():0.4f}')

decoder = keras.layers.StringLookup(vocabulary=encoder.get_vocabulary(), invert=True)
decode = lambda x: b''.join(decoder(x).numpy()).decode()


IPython.embed()