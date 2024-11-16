import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# タイトル
st.title("Fashion MNIST モデル推論")

# クラス名の定義
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# モデルのロード
@st.cache_resource
def load_trained_model():
    # 学習済みモデルをロード（事前にローカルでトレーニングして保存したもの）
    model_path = "fashion_mnist_model.h5"
    return load_model(model_path)

st.write("学習済みモデルをロードしています...")
model = load_trained_model()
st.success("学習済みモデルのロードが完了しました！")

# データセットのロード
@st.cache_resource
def load_data():
    from tensorflow.keras.datasets import fashion_mnist
    _, (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test / 255.0  # 正規化
    return x_test, y_test

x_test, y_test = load_data()

# モデルの推論
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
