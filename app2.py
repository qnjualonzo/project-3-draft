import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load Translation Model
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained("4yo1/llama3-eng-ko-8b-sl5")
    model = AutoModelForCausalLM.from_pretrained("4yo1/llama3-eng-ko-8b-sl5")
    return tokenizer, model

# Translate Text (using a generation approach with the causal language model)
def translate_text(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit App
st.title("English-Korean Translation with LLaMA")
st.write("This app translates text between English and Korean using the LLaMA model.")

# Input Section for Translation
st.header("English to Korean Translation")
text_to_translate = st.text_area("Enter text in English to translate to Korean:")

if st.button("Translate"):
    if text_to_translate:
        tokenizer, model = load_translation_model()
        translation = translate_text(text_to_translate, tokenizer, model)
        st.subheader("Translated Text:")
        st.write(translation)
    else:
        st.warning("Please enter text to translate.")

# Input Section for Summarization (you can keep the same T5-based summarization code)
st.header("Text Summarization")
text_to_summarize = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text_to_summarize:
        summary = summarize_text(text_to_summarize)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
