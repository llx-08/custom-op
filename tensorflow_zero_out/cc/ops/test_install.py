import tensorflow as tf
zero_out_module = tf.load_op_library('_zero_out_ops.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())