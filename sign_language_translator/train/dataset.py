import os
import pathlib
import random
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from train.frame_generator import FrameGenerator
import helpers.singword_db as swdb
import params

def create_dataset(db_path: str = params.WORDS_DB_PATH, dataset_path: str = params.DATASET_PATH) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    db = swdb.read_database(db_path)

    root_path = pathlib.Path(dataset_path)
    subset_paths = {}
    for split_name in os.listdir(root_path):
        split_dir = root_path / split_name
        subset_paths[split_name] = split_dir

    class_ids_for_name = {}

    for data in db:
        class_ids_for_name[data.zh_name] = data.class_id

    output_signature = (
        tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
        tf.TensorSpec(shape = (), dtype = tf.int16)
        )

    train_ds = tf.data.Dataset.from_generator(
        FrameGenerator(
            path = subset_paths['train'],
            num_frames = params.NUM_FRAMES,
            resolution = params.RESOLUTION,
            frame_step = params.FRAME_STEP,
            class_ids_for_name= class_ids_for_name,
            training = True,
            ),
        output_signature = output_signature
        )

    train_ds = train_ds.batch(params.BATCH_SIZE)

    val_ds = tf.data.Dataset.from_generator(
        FrameGenerator(
            path = subset_paths['val'],
            num_frames = params.NUM_FRAMES,
            resolution = params.RESOLUTION,
            frame_step = params.FRAME_STEP,
            class_ids_for_name = class_ids_for_name,
            ),
        output_signature = output_signature
        )

    val_ds = val_ds.batch(params.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_generator(
        FrameGenerator(
            path = subset_paths['test'],
            num_frames = params.NUM_FRAMES,
            resolution = params.RESOLUTION,
            frame_step = params.FRAME_STEP,
            class_ids_for_name= class_ids_for_name,
            ),
        output_signature = output_signature
        )

    test_ds = test_ds.batch(params.BATCH_SIZE)

    return train_ds, val_ds, test_ds