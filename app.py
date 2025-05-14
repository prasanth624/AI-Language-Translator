import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.markdown("<div style='font-size:40px; font-weight:bold;'>🌍 AI Language Translator</div>", unsafe_allow_html=True)

language_pairs = {
    "English to French": "Helsinki-NLP/opus-mt-en-fr",
    "English to Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English to German": "Helsinki-NLP/opus-mt-en-de",
    "English to Italian": "Helsinki-NLP/opus-mt-en-it",
    "English to Chinese": "Helsinki-NLP/opus-mt-en-zh",
    "English to Arabic": "Helsinki-NLP/opus-mt-en-ar",
    "English to Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "English to Swahili": "Helsinki-NLP/opus-mt-en-sw",
}

# Cache model/tokenizer loading
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Custom styled label for input
st.markdown("<div style='font-size:22px;'>📝 Enter text in Source Language</div>", unsafe_allow_html=True)
source_text = st.text_area("", "I love AI")

# Custom styled label for dropdown
st.markdown("<div style='font-size:22px;'>🌐 Choose Translation Pair</div>", unsafe_allow_html=True)
pair = st.selectbox("", list(language_pairs.keys()))
model_name = language_pairs[pair]

# Load model/tokenizer (cached)
tokenizer, model = load_model_and_tokenizer(model_name)

# Translation logic
if st.button("🚀 Translate"):
    if not source_text.strip():
        st.warning("⚠️ Please enter text to translate.")
    else:
        with st.spinner("🔄 Translating..."):
            # Split input by lines
            lines = source_text.strip().splitlines()
            translated_lines = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    inputs = tokenizer(line.strip(), return_tensors="pt")
                    translated_tokens = model.generate(**inputs)
                    translated_line = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    translated_lines.append(translated_line)
                else:
                    translated_lines.append("")  # Preserve blank lines

            # Join translated lines with line breaks
            translated_text = "\n".join(translated_lines)

        # Display translation
        st.markdown(
            f"<div style='font-size:18px; margin-top:20px;'>📄 Translated text in {pair.split(' to ')[1]}:</div>",
            unsafe_allow_html=True
        )
        st.text(translated_text)  # Use st.text to preserve line breaks and formatting
