import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import time

# ---------------- Stopword List ----------------
BUILTIN_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out over own same
shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've were weren't
what what's when when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# ----------------- Utility Functions ------------------

def pad_sequences(sequences, max_len, pad_token=0):
    return np.array([
        seq + [pad_token] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
        for seq in sequences
    ])

def create_vocab(texts, vocab_size=1000, remove_stopwords=False):
    all_words = [word for text in texts for word in text.split()]
    if remove_stopwords:
        all_words = [word for word in all_words if word.lower() not in BUILTIN_STOPWORDS]
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

def transformer(X, Y, vocab_size, d_model, num_heads, num_layers, show_step, remove_stopwords=False):
    show_step("🔡 Creating Vocabulary...")
    vocab = create_vocab(X + Y, vocab_size, remove_stopwords)
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
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Word Cloud of Input + Target Texts")
    st.pyplot(fig)

# ----------------- Streamlit App ------------------------

st.set_page_config(page_title="Transformer Visualizer", layout="wide")
st.title("🔍 Transformer Pipeline Visualizer")
st.markdown("Upload your Excel file with articles to visualize how a Transformer processes them.")

# Upload Excel file
uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        columns = df.columns.tolist()

        if "Input Text (X)" in columns and "Target Text (Y)" in columns:
            input_df = df[["Input Text (X)", "Target Text (Y)"]].copy()
        else:
            first_col = columns[0]
            input_df = pd.DataFrame()
            input_df["Input Text (X)"] = df[first_col].astype(str)
            input_df["Target Text (Y)"] = ["<empty>" for _ in range(len(df))]
            st.warning("⚠️ 'Target Text (Y)' not found — filled with placeholders.")
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        st.stop()
else:
    st.info("Please upload an Excel file with either a single column or 'Input Text (X)' and 'Target Text (Y)'")
    st.stop()

X_input = input_df["Input Text (X)"].fillna("").tolist()
Y_input = input_df["Target Text (Y)"].fillna("").tolist()

# Sidebar for model hyperparameters
with st.sidebar:
    st.header("⚙️ Transformer Settings")
    d_model = st.slider("Model Dimension", 32, 256, 128, step=32)
    num_heads = st.slider("Attention Heads", 1, 8, 4)
    num_layers = st.slider("Encoder/Decoder Layers", 1, 6, 2)
    vocab_size = st.slider("Vocabulary Size", 100, 2000, 1000, step=100)
    remove_stop = st.checkbox("Remove Stopwords from Vocabulary", value=True)

status = st.empty()

# Run Transformer
if st.button("🚀 Run Transformer"):
    def update_status(msg):
        status.info(msg)

    output, token_ids, vocab = transformer(
        X_input, Y_input, vocab_size, d_model, num_heads, num_layers, update_status, remove_stop
    )
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
