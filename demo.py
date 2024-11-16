import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go

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
    model_path = "fashion_mnist_model.h5"
    return tf.keras.models.load_model(model_path)

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

# 推論結果を表示する関数
def display_prediction(image, label, predictions):
    # 最も高い確率のクラスを取得
    predicted_class = np.argmax(predictions)
    predicted_prob = predictions[0][predicted_class]
    true_class_name = class_names[label] if label is not None else "N/A"
    predicted_class_name = class_names[predicted_class]

    # 結果の表示
    st.subheader("予測結果")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 入力画像")
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.markdown("### 予測情報")
        st.markdown(f"**予測クラス:** {predicted_class_name}")
        st.markdown(f"**正解クラス:** {true_class_name}")
        st.markdown(f"**確信度:** {predicted_prob:.2%}")

    # 確率をバーグラフで可視化
    st.markdown("### 各クラスの予測確率")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_names,
        y=predictions[0],
        text=[f"{p:.2%}" for p in predictions[0]],
        textposition='auto',
    ))
    fig.update_layout(
        xaxis_title="クラス",
        yaxis_title="確率",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig)

# モデルの推論
st.subheader("画像を選択またはアップロードして推論")

tab1, tab2 = st.tabs(["テストデータから選択", "画像をアップロード"])

with tab1:
    # サンプル画像の選択
    index = st.slider("画像を選択してください (0-9999)", 0, 9999, 0)
    sample_image = x_test[index]
    label = y_test[index]

    # 選択された画像を表示
    st.markdown("### 選択された画像はこちらです！")
    fig, ax = plt.subplots()
    ax.imshow(sample_image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    # 推論結果
    if st.button("この画像を推論してみる！", key='predict_sample'):
        predictions = model.predict(np.expand_dims(sample_image, axis=0))
        display_prediction(sample_image, label, predictions)

with tab2:
    # ファイルアップロード
    uploaded_file = st.file_uploader("ファッションアイテムの画像をアップロードしてください", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # 画像の読み込みと前処理
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0

        # アップロードされた画像を表示
        st.markdown("### アップロードされた画像はこちらです！")
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

        # 推論結果
        if st.button("この画像を推論してみる！", key='predict_uploaded'):
            predictions = model.predict(np.expand_dims(image_array, axis=0))
            display_prediction(image_array, None, predictions)
