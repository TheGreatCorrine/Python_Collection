import sys
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except:
    print("TensorFlow not installed properly")

try:
    print(f"TF Keras version: {tf.keras.__version__}")
except:
    print("Cannot access tf.keras")

# import tensorflow as tf
# model = tf.keras.Sequential()
# print("基本导入成功")

import tensorflow as tf

# 使用Input层作为第一层
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),  # 替换成您的实际输入维度
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])