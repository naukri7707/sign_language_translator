import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

print("------------------------")
print(f"TensorFlow version: {tf.__version__}")
if gpus:
    print(len(gpus), "GPUs detected!")
else:
    print("No GPUs detected.")
    
print("------------------------")