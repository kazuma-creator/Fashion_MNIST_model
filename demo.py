import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# タイトル
st.title("Fashion MNIST モデル作成と推論")

# データセットのロード
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# データの前処理
x_train, x_test = x_train / 255.0, x_test / 255.0

# モデルの定義
def create_model():
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
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# モデルの作成と学習
st.write("モデルを作成中...しばらくお待ちください")
model = create_model()
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
st.success("モデルの学習が完了しました！")
st.session_state['model'] = model  # 学習済みモデルを保存

# モデルの推論
if 'model' in st.session_state:
    model = st.session_state['model']

    st.subheader("画像を選択またはアップロードして推論")

    tab1, tab2 = st.tabs(["テストデータから選択", "画像をアップロード"])

    with tab1:
        # サンプル画像の選択
        index = st.slider("画像を選択してください (0-9999)", 0, 9999, 0)
        sample_image = x_test[index]
        label = y_test[index]

        # 選択された画像を表示
        st.write("選択された画像はこちらです！")
        fig, ax = plt.subplots()
        ax.imshow(sample_image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

        # 推論結果
        if st.button("この画像を推論してみる！", key='predict_sample'):
            predictions = model.predict(np.expand_dims(sample_image, axis=0))
            predicted_class = np.argmax(predictions)
            st.write(f"予測結果: **{class_names[predicted_class]}** (正解: **{class_names[label]}**)")
            st.write("各クラスの予測確率:")
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {predictions[0][i]:.2%}")

    with tab2:
        # ファイルアップロード
        uploaded_file = st.file_uploader("ファッションアイテムの画像をアップロードしてください", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # 画像の読み込みと前処理
            image = Image.open(uploaded_file).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image) / 255.0

            # 画像を表示
            st.write("アップロードされた画像はこちらです！")
            fig, ax = plt.subplots()
            ax.imshow(image_array, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

            # 推論結果
            if st.button("この画像を推論してみる！", key='predict_uploaded'):
                predictions = model.predict(np.expand_dims(image_array, axis=0))
                predicted_class = np.argmax(predictions)
                st.write(f"予測結果: **{class_names[predicted_class]}**")
                st.write("各クラスの予測確率:")
                for i, class_name in enumerate(class_names):
                    st.write(f"{class_name}: {predictions[0][i]:.2%}")
