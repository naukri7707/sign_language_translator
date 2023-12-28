import tensorflow as tf

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

import train.dataset as ds
import params

train_ds, val_ds, test_ds = ds.create_dataset()

model_id = params.MOVINET_MODEL_ID
model_mode = 'base'

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

checkpoint_dir = f'movinet_{model_id}_{model_mode}'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes, freeze_backbone=False):
    """Builds a classifier on top of a backbone model."""
    res_x, res_y = resolution
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate = params.DROPOUT_RATE,
        )
    model.build([batch_size, num_frames, res_x, res_y, 3])

    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True

    return model

model = build_classifier(
    params.BATCH_SIZE,
    params.NUM_FRAMES,
    params.RESOLUTION,
    backbone,
    params.NUM_CLASSES
    )

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate = params.LEARNING_RATE)
model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = "ckpt/sign_language-ep={epoch:02d}-ls={val_loss:.2f}-ac={val_accuracy:.2f}.hdf5",
    save_weights_only = True,
    monitor = 'val_accuracy',
    mode='max',
    save_best_only=True)

callbacks = [
    model_checkpoint_callback,
    tf.keras.callbacks.TensorBoard(),
]

results = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = params.EPOCHS,
    validation_freq = 1,
    verbose = 1,
    callbacks=callbacks
    )