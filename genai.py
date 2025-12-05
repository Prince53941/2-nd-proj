import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Gen AI with Transformers",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Gen AI with Transformer Library")
st.write(
    "Interactive demo of multiple NLP tasks using Hugging Face Transformers "
    "and SentenceTransformers."
)

# ---------------------------
# Cached model loaders
# ---------------------------

@st.cache_resource
def load_text_generation():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_ner():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

@st.cache_resource
def load_qa():
    return pipeline("question-answering")

@st.cache_resource
def load_translation():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

@st.cache_resource
def load_paraphraser():
    return pipeline("text2text-generation", model="t5-small")

@st.cache_resource
def load_grammar_corrector():
    return pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")

@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# Sidebar: Task selection
# ---------------------------

task = st.sidebar.selectbox(
    "Choose a task",
    [
        "Text Generation",
        "Summarization",
        "Sentiment Analysis",
        "Named Entity Recognition (NER)",
        "Question Answering",
        "Translation (EN → FR)",
        "Paraphrasing",
        "Keywords via NER",
        "Grammar Correction",
        "Text Similarity"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("All tasks run locally using Hugging Face pipelines.")


# ---------------------------
# Task: Text Generation
# ---------------------------
if task == "Text Generation":
    st.header("📝 Text Generation (GPT-2)")
    default_prompt = "AI will change the world because"
    prompt = st.text_area("Enter a prompt:", value=default_prompt, height=120)
    max_len = st.slider("Max length", min_value=20, max_value=400, value=100, step=10)
    do_sample = st.checkbox("Use sampling (more creative)", value=True)

    if st.button("Generate"):
        with st.spinner("Generating text..."):
            generator = load_text_generation()
            result = generator(
                prompt,
                max_length=max_len,
                do_sample=do_sample,
                num_return_sequences=1
            )
        st.subheader("Output")
        st.write(result[0]["generated_text"])


# ---------------------------
# Task: Summarization
# ---------------------------
elif task == "Summarization":
    st.header("📚 Summarization (BART)")
    default_text = (
        "Artificial intelligence is transforming industries worldwide. "
        "Companies use AI to automate processes, gain insights from data, "
        "and improve decision-making. As AI continues to advance, its impact "
        "on society will grow significantly."
    )
    text = st.text_area("Enter text to summarize:", value=default_text, height=200)
    max_len = st.slider("Max summary length", 20, 150, 50, step=5)
    min_len = st.slider("Min summary length", 5, 80, 10, step=5)

    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summarizer = load_summarizer()
            summary = summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
        st.subheader("Summary")
        st.write(summary[0]["summary_text"])


# ---------------------------
# Task: Sentiment Analysis
# ---------------------------
elif task == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    default_text = "I love using machine learning tools—they make life easier!"
    text = st.text_area("Enter text:", value=default_text, height=150)

    if st.button("Analyze sentiment"):
        with st.spinner("Analyzing sentiment..."):
            sentiment_model = load_sentiment()
            result = sentiment_model(text)
        st.subheader("Result")
        st.json(result)


# ---------------------------
# Task: Named Entity Recognition (NER)
# ---------------------------
elif task == "Named Entity Recognition (NER)":
    st.header("🏷️ Named Entity Recognition (NER)")
    default_text = "Elon Musk founded SpaceX in California."
    text = st.text_area("Enter text:", value=default_text, height=150)

    if st.button("Run NER"):
        with st.spinner("Running NER..."):
            ner = load_ner()
            entities = ner(text)
        st.subheader("Entities")
        st.json(entities)


# ---------------------------
# Task: Question Answering
# ---------------------------
elif task == "Question Answering":
    st.header("❓ Question Answering")
    default_context = "The Taj Mahal is located in Agra, India."
    context = st.text_area("Context:", value=default_context, height=150)
    question = st.text_input("Question:", value="Where is Taj Mahal?")

    if st.button("Get answer"):
        with st.spinner("Answering..."):
            qa = load_qa()
            result = qa(question=question, context=context)
        st.subheader("Answer")
        st.write(result["answer"])
        st.write(f"Score: {result['score']:.3f}")


# ---------------------------
# Task: Translation (EN → FR)
# ---------------------------
elif task == "Translation (EN → FR)":
    st.header("🌍 Translation (English → French)")
    text = st.text_area("Enter English text:", value="How are you?", height=120)

    if st.button("Translate"):
        with st.spinner("Translating..."):
            translate = load_translation()
            out = translate(text)
        st.subheader("French Translation")
        st.write(out[0]["translation_text"])


# ---------------------------
# Task: Paraphrasing
# ---------------------------
elif task == "Paraphrasing":
    st.header("🔁 Paraphrasing (T5-small)")
    default_text = "Machine learning is interesting."
    text = st.text_area("Enter text to paraphrase:", value=default_text, height=150)

    if st.button("Paraphrase"):
        with st.spinner("Paraphrasing..."):
            para = load_paraphraser()
            out = para(f"paraphrase: {text}")
        st.subheader("Paraphrased Text")
        st.write(out[0]["generated_text"])


# ---------------------------
# Task: Keywords via NER
# ---------------------------
elif task == "Keywords via NER":
    st.header("🔑 Keyword Extraction via NER")
    default_text = "Amazon is investing $10 billion to expand cloud services in India."
    text = st.text_area("Enter text:", value=default_text, height=150)

    if st.button("Extract keywords (entities)"):
        with st.spinner("Extracting entities..."):
            ner = load_ner()
            entities = ner(text)
        st.subheader("Entities (as keywords)")
        st.json(entities)


# ---------------------------
# Task: Grammar Correction
# ---------------------------
elif task == "Grammar Correction":
    st.header("✍️ Grammar Correction")
    default_text = "She go to school every days"
    text = st.text_area("Enter sentence:", value=default_text, height=120)

    if st.button("Correct grammar"):
        with st.spinner("Correcting grammar..."):
            gc = load_grammar_corrector()
            out = gc(text)
        st.subheader("Corrected Sentence")
        st.write(out[0]["generated_text"])


# ---------------------------
# Task: Text Similarity
# ---------------------------
elif task == "Text Similarity":
    st.header("📏 Text Similarity (Sentence Transformers)")
    text_a = st.text_area(
        "Sentence A:",
        value="AI will change the world",
        height=100
    )
    text_b = st.text_area(
        "Sentence B:",
        value="Artificial intelligence will transform industries",
        height=100
    )

    if st.button("Compute similarity"):
        with st.spinner("Computing similarity..."):
            model = load_similarity_model()
            a = model.encode(text_a, convert_to_tensor=True)
            b = model.encode(text_b, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(a, b).item()
        st.subheader("Cosine Similarity")
        st.write(f"{sim:.4f}")
