import tensorflow as tf
import keras

import r2plus1d.networks as net
from r2plus1d.frame_generator import FrameGenerator

EPOCHS = 50
N_FRAMES = 20
BATCH_SIZE = 64
IMG_HEIGHT = 360
IMG_WIDTH = 360
CLASS_NUM = 10

output_signature = (
    tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.int16)
    )

train_ds = tf.data.Dataset.from_generator(
    FrameGenerator('~data/1~10', N_FRAMES, glob_patten='*/*_offset1.mp4', training=True),
    output_signature = output_signature
    )


# Batch the data
train_ds = train_ds.batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_generator(
    FrameGenerator('~data/1~10', N_FRAMES, glob_patten='*/*_offset2.mp4',),
    output_signature = output_signature
    )

val_ds = val_ds.batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_generator(
    FrameGenerator('~data/1~10', N_FRAMES, glob_patten='*/*_offset3.mp4',),
    output_signature = output_signature
    )

test_ds = test_ds.batch(BATCH_SIZE)

model = net.create(IMG_HEIGHT, IMG_WIDTH, CLASS_NUM)

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
    epochs = EPOCHS,
    validation_data = val_ds,
    callbacks=[model_checkpoint_callback]
    )