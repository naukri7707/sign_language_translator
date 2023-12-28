import tensorflow as tf
import keras

import r2plus1d as net
from train.frame_generator import FrameGenerator
import params

output_signature = (
    tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.int16)
    )

dataset_path = "~data/dataset/src_1~10_flow_split"

train_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        f'{dataset_path}/train',
        params.NUM_FRAMES,
        params.FRAME_SIZE,
        params.FRAME_STEP,
        training=True,
        ),
    output_signature = output_signature
    )

# Batch the data
train_ds = train_ds.batch(params.BATCH_SIZE)

val_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        f'{dataset_path}/val',
        params.NUM_FRAMES,
        params.FRAME_SIZE,
        params.FRAME_STEP,
        ),
    output_signature = output_signature
    )

val_ds = val_ds.batch(params.BATCH_SIZE)

test_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        f'{dataset_path}/test',
        params.NUM_FRAMES,
        params.FRAME_SIZE,
        params.FRAME_STEP,
        ),
    output_signature = output_signature
    )

test_ds = test_ds.batch(params.BATCH_SIZE)

frame_x, frame_y = params.FRAME_SIZE

model = net.create(frame_x, frame_y, params.NUM_CLASSES)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    metrics = ['accuracy']
    )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = "ckpt/sign_language-ep={epoch:02d}-ls={val_loss:.2f}-ac={val_accuracy:.2f}.hdf5",
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