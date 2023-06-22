import tensorflow as tf
import matplotlib.pyplot as plt
import os, PIL, pathlib,glob
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras import Model

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]], "GPU")


data_dir = './bird/bird_photos'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*')))
print("图片总数为: ", image_count)

# 加载数据
batch_size = 8
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 5))

for images, labels in train_ds.take(1):
    for i in range(8):

        ax = plt.subplot(2, 4, i+1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])

        plt.axis("off")


# 再次检查数据
for image_batch, labels_batch in train_ds:

    print(image_batch.shape)
    print(labels_batch.shape)
    break


# 配置数据集
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# 构建resnet-50网络中的残差模块
def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters_1, filters_2, filters_3 = filters

    name_base = str(stage) + block + '_identity_block_'

    x = Conv2D(filters_1, (1, 1), name=name_base+'conv1')(input_tensor)
    x = BatchNormalization(name=name_base + 'bn1')(x)
    x = Activation('relu', name=name_base + 'relu1')(x)

    x = Conv2D(filters_2, kernel_size, padding='same', name=name_base + 'conv2')(x)
    x = BatchNormalization(name=name_base + 'bn2')(x)
    x = Activation('relu', name=name_base + 'relu2')(x)

    x = Conv2D(filters_3, (1, 1), name=name_base + 'conv3')(x)
    x = BatchNormalization(name=name_base + 'bn3')(x)

    x = layers.add([x, input_tensor], name=name_base + 'add')
    x = Activation('relu', name=name_base + 'relu4')(x)

    return x


# 卷积模块
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters_1, filters_2, filters_3 = filters

    res_name_base = str(stage) + block + '_conv_block_res_'
    name_base = str(stage) + block + '_conv_block_'

    x = Conv2D(filters_1, (1, 1), strides=strides, name=name_base + 'conv1')(input_tensor)
    x = BatchNormalization(name=name_base + 'bn1')(x)
    x = Activation('relu', name=name_base + 'relu1')(x)

    x = Conv2D(filters_2, kernel_size, padding='same', name=name_base + 'conv2')(x)
    x = BatchNormalization(name=name_base + 'bn2')(x)
    x = Activation('relu', name=name_base + 'relu2')(x)

    x = Conv2D(filters_3, (1, 1), name=name_base + 'conv3')(x)
    x = BatchNormalization(name=name_base + 'bn3')(x)

    shortcut = Conv2D(filters_3, (1, 1), strides=strides, name=res_name_base + 'conv')(input_tensor)
    shortcut = BatchNormalization(name=res_name_base + 'bn')(shortcut)

    x = layers.add([x, shortcut], name=name_base + 'add')
    x = Activation('relu', name=name_base + 'relu4')(x)

    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    # 加载预训练模型
    model.load_weights('./bird/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    return model


model = ResNet50()
print(model.summary())


# 编译
opt = tf.keras.optimizers.Adam(learning_rate=1e-7)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )


# 模型训练
epochs = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.show()
























