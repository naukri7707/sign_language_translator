# Edit from https://github.com/tensorflow/models/blob/master/official/projects/movinet/movinet_streaming_model_training_and_inference.ipynb

import tensorflow as tf
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model

import train.dataset as ds
import params

train_ds, val_ds, test_ds = ds.create_dataset()

model_id = params.MOVINET_MODEL_ID
model_mode = 'stream'

for frames, labels in train_ds.take(1):
  print(f"Shape: {frames.shape}")
  print(f"Label: {labels.shape}")

use_positional_encoding = model_id in {'a3', 'a4', 'a5'}

backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=False,
)

# Note: this is a temporary model constructed for the
# purpose of loading the pre-trained checkpoint. Only
# the backbone will be used to build the custom classifier.
model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=600,
    output_states=True,
    )

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([1, 13, 172, 172, 3])

# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)

checkpoint_dir = f'movinet_{model_id}_{model_mode}'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

# Detect hardware
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
    tpu_resolver = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

# Select appropriate distribution strategy
if tpu_resolver:
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
    print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    distribution_strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    distribution_strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    distribution_strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')

print("Number of accelerators: ", distribution_strategy.num_replicas_in_sync)

## 使用所需數量的類構建自定義分類器

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes, freeze_backbone=False):
    """Builds a classifier on top of a backbone model."""
    res_x, res_y = resolution
    model = movinet_model.MovinetClassifier(
        backbone = backbone,
        num_classes = num_classes,
        dropout_rate = params.DROPOUT_RATE,
        )
    model.build([batch_size, num_frames, res_x, res_y, 3])

    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True

    return model

# Construct loss, optimizer and compile the model
with distribution_strategy.scope():
    model = build_classifier(
        params.BATCH_SIZE,
        params.NUM_FRAMES, 
        params.RESOLUTION,
        backbone,
        params.NUM_CLASSES,
        freeze_backbone = True # 凍結預訓練模型權重
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
    validation_data=val_ds,
    epochs=params.EPOCHS,
    validation_freq=1,
    verbose=1,
    callbacks=callbacks
    )