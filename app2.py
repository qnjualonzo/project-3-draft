import streamlit as st
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer


# Load Translation Pipelines for English-Korean and Korean-English
def load_translation_pipeline(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-tc-big-{src_lang}-{tgt_lang}"
    pipe = pipeline("translation", model=model_name)
    return pipe


# Translate Text
def translate_text(text, src_lang, tgt_lang):
    pipe = load_translation_pipeline(src_lang, tgt_lang)
    translation = pipe(text)
    return translation[0]['translation_text']


# Load Summarization Model
@st.cache_resource
def load_summarization_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


# Summarize Text
def summarize_text(text):
    tokenizer, model = load_summarization_model()
    input_ids = tokenizer.encode(
        "summarize: " + text, return_tensors="pt", max_length=512, truncation=True
    )
    summary_ids = model.generate(
        input_ids,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Streamlit App
st.title("Machine Translation and Summarization App")
st.write(
    "This app provides free text translation (Korean to English and vice-versa) and summarization using open-source models."
)

# Translation Section
st.header("Translation")
text_to_translate = st.text_area("Enter text to translate:")
src_lang = st.selectbox("Source Language", ["en", "ko"], help="Select the source language.")
tgt_lang = st.selectbox("Target Language", ["ko", "en"], help="Select the target language.")

if st.button("Translate"):
    if text_to_translate:
        if src_lang != tgt_lang:
            translation = translate_text(text_to_translate, src_lang, tgt_lang)
            st.subheader("Translated Text:")
            st.write(translation)
        else:
            st.warning("Please choose different source and target languages.")
    else:
        st.warning("Please enter text to translate.")

# Summarization Section
st.header("Text Summarization")
text_to_summarize = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text_to_summarize:
        summary = summarize_text(text_to_summarize)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
