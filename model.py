from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets import fashion_mnist

# データセットのロード
(x_train, y_train), _ = fashion_mnist.load_data()
x_train = x_train / 255.0

# モデル定義
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# トレーニング
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# モデルの保存
model.save("fashion_mnist_model.h5")
