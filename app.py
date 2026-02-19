import os
import streamlit as st
import pdfplumber
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Disabilita il warning sui symlink di Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Cache del modello per evitare di caricarlo ogni volta
@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# Titolo dell'app
st.title("PDF Concept Extractor 💡")
st.write("Carica un PDF e ottieni i concetti principali!")

# Upload del file
uploaded_file = st.file_uploader("Seleziona un PDF", type="pdf")

if uploaded_file:
    # Estrazione testo dal PDF !!
    testo = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for pagina in pdf.pages:
            pagina_testo = pagina.extract_text()
            if pagina_testo:
                testo += pagina_testo + "\n"

    if testo.strip() == "":
        st.error("Non è stato possibile estrarre testo dal PDF.")
    else:
        st.subheader("📄 Testo estratto (anteprima)")
        st.write(testo[:1000] + "...")  # anteprima dei primi 1000 caratteri

        # Riassunto con Hugging Face Transformers
        st.subheader("📝 Concetti principali (riassunto)")
        
        try:
            model, tokenizer = load_summarizer()
            
            # Dividi il testo in chunk più grandi (per parole, non caratteri)
            parole = testo.split()
            max_words = 200  # ~1024 token per chunk
            riassunto_completo = ""
            
            for i in range(0, len(parole), max_words):
                chunk = " ".join(parole[i:i+max_words])
                if len(chunk.split()) > 50:  # Solo se il chunk ha almeno 50 parole
                    try:
                        # Tokenizza il testo
                        inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)
                        
                        # Genera il riassunto
                        summary_ids = model.generate(
                            inputs,
                            max_length=1000000,
                            min_length=100,
                            num_beams=4,
                            length_penalty=2.0,
                            early_stopping=True
                        )
                        
                        # Decodifica il riassunto
                        riassunto = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        riassunto_completo += riassunto + " "
                    except:
                        continue
            
            st.write(riassunto_completo.strip() if riassunto_completo.strip() else "Impossibile generare un riassunto.")
        except Exception as e:
            st.error(f"Errore durante la generazione del riassunto: {str(e)}")
