
import train.dataset as ds
import params
from preprocess import split_data

split_data.split('~data/dataset/src_hand', train=8, val=1, test=1)

# train_ds, val_ds, test_ds = ds.create_dataset()