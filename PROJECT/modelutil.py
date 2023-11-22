import tensorflow as tf
from tensorflow.keras.models import load_model

def CTCLoss(y_true,y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1],dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1],dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def load_model1():
    # Load the entire model from the HDF5 file
    model = load_model(r"C:\Users\banou\Downloads\LipReadingTrainedModel.h5", custom_objects={'CTCLoss': CTCLoss})

    return model