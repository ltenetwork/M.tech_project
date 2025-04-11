import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import time

# ----------------- Utility Functions ------------------

def pad_sequences(sequences, max_len, pad_token=0):
    return np.array([
        seq + [pad_token] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
        for seq in sequences
    ])

def create_vocab(texts, vocab_size=1000):
    all_words = [word for text in texts for word in text.split()]
    most_common_words = [word for word, _ in Counter(all_words).most_common(vocab_size - 1)]
    vocab = {word: idx + 1 for idx, word in enumerate(most_common_words)}
    vocab["<UNK>"] = 0
    return vocab

def encode_text(text, vocab):
    return [vocab.get(word, 0) for word in text.split()]

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

def transformer_encoder(X, *args, **kwargs):
    return X + np.random.normal(0, 0.1, size=X.shape)

def transformer_decoder(Y, encoder_output, num_heads):
    return Y + np.random.normal(0, 0.1, size=Y.shape)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# --------------- Transformer Pipeline ------------------

def transformer(X, Y, vocab_size, d_model, num_heads, num_layers, show_step):
    show_step("🔡 Creating Vocabulary...")
    vocab = create_vocab(X + Y, vocab_size)
    time.sleep(0.5)

    show_step("🔢 Encoding Texts...")
    X_ids = [encode_text(x, vocab) for x in X]
    Y_ids = [encode_text(y, vocab) for y in Y]
    time.sleep(0.5)

    show_step("📏 Padding Sequences...")
    max_len = max(max(map(len, X_ids)), max(map(len, Y_ids)))
    X_ids = pad_sequences(X_ids, max_len)
    Y_ids = pad_sequences(Y_ids, max_len)
    time.sleep(0.5)

    batch_size, seq_len = X_ids.shape

    show_step("📐 Creating Embeddings...")
    embedding_matrix = np.random.randn(vocab_size, d_model)
    X_enc = embedding_matrix[X_ids]
    Y_dec = embedding_matrix[Y_ids]

    PE = positional_encoding(seq_len, d_model)
    X_enc += PE
    Y_dec += PE
    time.sleep(0.5)

    show_step("🧠 Encoding...")
    for i in range(num_layers):
        show_step(f"➡️ Encoder Layer {i+1}")
        X_enc = transformer_encoder(X_enc)
        time.sleep(0.3)

    show_step("🧠 Decoding...")
    for i in range(num_layers):
        show_step(f"➡️ Decoder Layer {i+1}")
        Y_dec = transformer_decoder(Y_dec, X_enc, num_heads)
        time.sleep(0.3)

    show_step("🎯 Output Softmax Projection...")
    final_weights = np.random.randn(d_model, vocab_size)
    logits = np.dot(Y_dec, final_weights)
    output = softmax(logits)

    predicted_ids = np.argmax(output, axis=-1)
    inv_vocab = {idx: word for word, idx in vocab.items()}
    decoded_texts = []
    for seq in predicted_ids:
        decoded_seq = [inv_vocab.get(idx, "<UNK>") for idx in seq]
        decoded_texts.append(" ".join(decoded_seq))
    return decoded_texts, X_ids, vocab

# ----------------- Graph Utilities ------------------

def plot_token_distribution(texts):
    words = [word for text in texts for word in text.split()]
    freq = Counter(words)
    top_words = freq.most_common(15)
    df = pd.DataFrame(top_words, columns=["Token", "Frequency"])
    fig, ax = plt.subplots()
    sns.barplot(x="Frequency", y="Token", data=df, palette="viridis", ax=ax)
    ax.set_title("Top 15 Token Frequencies")
    st.pyplot(fig)

def plot_sequence_lengths(token_ids, title="Sequence Lengths"):
    lengths = [len([token for token in seq if token != 0]) for seq in token_ids]
    fig, ax = plt.subplots()
    sns.histplot(lengths, kde=False, bins=10, ax=ax, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("Length")
    st.pyplot(fig)

def plot_attention_heatmap():
    fig, ax = plt.subplots()
    attention = np.random.rand(10, 10)
    sns.heatmap(attention, cmap="YlGnBu", ax=ax, cbar=True)
    ax.set_title("Simulated Attention Heatmap")
    st.pyplot(fig)

def plot_multi_attention_maps(n=3):
    fig, axs = plt.subplots(1, n, figsize=(n * 4, 4))
    for i in range(n):
        attention = np.random.rand(10, 10)
        sns.heatmap(attention, cmap="coolwarm", ax=axs[i], cbar=i == n - 1)
        axs[i].set_title(f"Head {i+1}")
    st.pyplot(fig)

def plot_word_cloud(texts):
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Word Cloud of Input + Target Texts")
    st.pyplot(fig)

# ----------------- Streamlit App ------------------------

st.set_page_config(page_title="Transformer Visualizer", layout="wide")

st.title("🔍 Transformer Pipeline Visualizer")
st.markdown("This app simulates a basic Transformer encoder-decoder architecture and visualizes components.")

# Input Data
st.subheader("📥 Input Data")
default_data = [
    {"Input Text (X)": "the sky is blue", "Target Text (Y)": "le ciel est bleu"},
    {"Input Text (X)": "hello world", "Target Text (Y)": "bonjour le monde"}
]
input_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

# Convert list of dicts to DataFrame if needed
if isinstance(input_df, list):
    input_df = pd.DataFrame(input_df)

X_input = input_df["Input Text (X)"].fillna("").tolist()
Y_input = input_df["Target Text (Y)"].fillna("").tolist()

# Sidebar for model hyperparameters
with st.sidebar:
    st.header("⚙️ Transformer Settings")
    d_model = st.slider("Model Dimension", 32, 256, 128, step=32)
    num_heads = st.slider("Attention Heads", 1, 8, 4)
    num_layers = st.slider("Encoder/Decoder Layers", 1, 6, 2)
    vocab_size = st.slider("Vocabulary Size", 100, 2000, 1000, step=100)

status = st.empty()

# Run Transformer
if st.button("🚀 Run Transformer"):
    def update_status(msg):
        status.info(msg)

    output, token_ids, vocab = transformer(X_input, Y_input, vocab_size, d_model, num_heads, num_layers, update_status)
    st.success("✅ Transformer Finished!")

    st.subheader("📘 Output Sequences:")
    for i, seq in enumerate(output):
        st.write(f"**Output {i+1}:** {seq}")

    st.markdown("---")
    st.subheader("📊 Token Analysis")
    col1, col2 = st.columns(2)
    with col1:
        plot_token_distribution(X_input + Y_input)
    with col2:
        plot_sequence_lengths(token_ids, "Input Sequence Lengths")

    st.subheader("🌥️ Word Cloud")
    plot_word_cloud(X_input + Y_input)

    st.subheader("🧠 Attention Simulation")
    plot_attention_heatmap()

    st.subheader("🎯 Multi-Head Attention Map")
    plot_multi_attention_maps(3)
