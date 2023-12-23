import tensorflow as tf
import keras

import r2plus1d as net
from r2plus1d.frame_generator import FrameGenerator
import params

output_signature = (
    tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.int16)
    )

train_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        '~data/1~10',
        params.N_FRAMES,
        params.FRAME_SIZE,
        params.FRAME_STEP,
        glob_patten='*/*_offset1.mp4',
        training=True,
        ),
    output_signature = output_signature
    )

# Batch the data
train_ds = train_ds.batch(params.BATCH_SIZE)

val_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        '~data/1~10',
        params.N_FRAMES,
        params.FRAME_SIZE,
        params.FRAME_STEP,
        glob_patten='*/*_offset2.mp4',
        ),
    output_signature = output_signature
    )

val_ds = val_ds.batch(params.BATCH_SIZE)

test_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        '~data/1~10',
        params.N_FRAMES,
        params.FRAME_SIZE,
        params.FRAME_STEP,
        glob_patten='*/*_offset3.mp4',
        ),
    output_signature = output_signature
    )

test_ds = test_ds.batch(params.BATCH_SIZE)

frame_x, frame_y = params.FRAME_SIZE

model = net.create(frame_x, frame_y, params.CLASS_NUM)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
    metrics = ['accuracy']
    )

checkpoint_filepath = 'ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
    save_weights_only = True,
    monitor = 'val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
    x = train_ds,
    epochs = params.EPOCHS,
    validation_data = val_ds,
    callbacks=[model_checkpoint_callback]
    )