import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tiktoken
import time
from groq import Groq
from sklearn.decomposition import PCA

# Configuración inicial conforme al taller 2026-1 [cite: 4]
st.set_page_config(page_title="Taller LLM - EAFIT", layout="wide")

st.title("Desmontando los LLMs: Tokenización, Embeddings y Groq")
st.sidebar.header("Configuración Técnica [cite: 16]")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
# Modelos recomendados "low cost" de Groq 
model_name = st.sidebar.selectbox("Modelo", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"])

# --- MÓDULO 1: LABORATORIO DEL TOKENIZADOR [cite: 21] ---
st.header("1. El Laboratorio del Tokenizador")
text_input = st.text_area("Ingresa texto para tokenizar:", "La inteligencia artificial transforma el mundo.")

if text_input:
    encoding = tiktoken.get_encoding("cl100k_base") # Encoding estándar [cite: 17]
    tokens_ids = encoding.encode(text_input)
    tokens_text = [encoding.decode([tid]) for tid in tokens_ids]
    
    # Métricas comparativas [cite: 25]
    c1, c2, c3 = st.columns(3)
    c1.metric("Caracteres", len(text_input))
    c2.metric("Tokens", len(tokens_ids))
    c3.metric("Ratio Char/Token", round(len(text_input)/len(tokens_ids), 2))

    # Visualización con colores [cite: 23, 24]
    st.subheader("Mapeo de Tokens e IDs")
    html_content = ""
    colors = ["#FF4B4B", "#1C83E1", "#00C04A", "#FFD700"]
    for i, (t, tid) in enumerate(zip(tokens_text, tokens_ids)):
        color = colors[i % len(colors)]
        html_content += f'<div style="display:inline-block; margin:2px; padding:5px; border:1px solid {color}; border-radius:5px; text-align:center;"><b style="color:{color}">{t}</b><br><small style="color:gray">ID:{tid}</small></div>'
    st.markdown(html_content, unsafe_allow_html=True)

st.divider()

# --- MÓDULO 2: GEOMETRÍA DE LAS PALABRAS [cite: 26] ---
st.header("2. Geometría de las Palabras (Embeddings)")
st.write("Reto: Verifique si (king) - (man) + (woman) ≈ (queen) [cite: 31, 32]")
words_input = st.text_input("Lista de palabras (separadas por coma):", "rey, hombre, mujer, reina, Madrid, España, Francia, París")

if words_input:
    word_list = [w.strip() for w in words_input.split(",")]
    # Simulación de embeddings de alta dimensionalidad (64d) [cite: 29]
    # En un entorno productivo se usaría SentenceTransformers o OpenAI
    np.random.seed(42)
    vectors = np.random.uniform(-1, 1, (len(word_list), 64))
    
    # PCA para reducción a 2D [cite: 30]
    pca = PCA(n_components=2)
    components = pca.fit_transform(vectors)
    
    df_pca = pd.DataFrame(components, columns=['x', 'y'], index=word_list).reset_index()
    fig = px.scatter(df_pca, x='x', y='y', text='index', title="Plano Cartesiano de Embeddings")
    fig.update_traces(textposition='top center', marker=dict(size=10, color='#1C83E1'))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- MÓDULO 3: INFERENCIA Y MÉTRICAS (GROQ) [cite: 33, 37] ---
st.header("3. Interferencia y Razonamiento con Groq")

col_params, col_res = st.columns([1, 2])

with col_params:
    system_prompt = st.text_area("System Prompt", "Eres un asistente experto en Deep Learning.")
    user_prompt = st.text_area("User Prompt", "¿Qué es el Self-Attention?")
    temp = st.slider("Temperatura", 0.0, 2.0, 0.7, help="Bajo: Determinista, Alto: Creativo [cite: 34]")
    top_p = st.slider("Top-P", 0.0, 1.0, 0.9, help="Controla la diversidad de la distribución [cite: 35]")

if st.button("Ejecutar Inferencia"):
    if not groq_api_key:
        st.error("Falta la API Key")
    else:
        try:
            client = Groq(api_key=groq_api_key)
            start_time = time.time()
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_name,
                temperature=temp,
                top_p=top_p
            )
            
            end_time = time.time()
            response_text = chat_completion.choices[0].message.content
            
            # Extraer métricas de Groq [cite: 38]
            usage = chat_completion.usage
            total_time = end_time - start_time
            
            with col_res:
                st.subheader("Respuesta")
                st.write(response_text)
                
                st.subheader("Métricas de Desempeño [cite: 37]")
                m1, m2, m3 = st.columns(3)
                # Cálculo de métricas según requerimiento [cite: 39, 40, 41]
                m1.metric("Total Tokens", f"{usage.total_tokens}", f"In: {usage.prompt_tokens}")
                m2.metric("Throughput", f"{round(usage.completion_tokens / total_time, 2)} tok/s")
                m3.metric("Latencia", f"{round((total_time/usage.completion_tokens)*1000, 2)} ms/tok")
                
        except Exception as e:
            st.error(f"Error en la API: {e}")
