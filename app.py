import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq
from sklearn.decomposition import PCA

# Configuración de la página
st.set_page_config(page_title="LLM Explorer: Tokens & Geometry", layout="wide")

st.title("Explore la Geometría de los LLMs 🧠")
st.markdown("""
Esta aplicación permite visualizar cómo un modelo de lenguaje procesa el texto, 
desde su fragmentación en tokens hasta su representación en un espacio vectorial.
""")

# --- SIDEBAR: Configuración y API Key ---
with st.sidebar:
    st.header("Configuración")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Introduce tu clave de Groq Cloud")
    
    model_option = st.selectbox(
        "Selecciona el Modelo",
        [
            "llama-3.1-8b-instant",   # El sucesor directo y rápido
            "llama-3.3-70b-versatile", # Muy potente y equilibrado
            "llama-3.1-70b-versatile", 
            "mixtral-8x7b-32768"
        ]
    )
    
    st.subheader("Hiperparámetros de Inferencia")
    temp = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    max_tokens = st.slider("Max Tokens", 1, 4096, 1024)

# --- LÓGICA DE PROCESAMIENTO ---
if not groq_api_key:
    st.warning("Por favor, introduce tu API Key de Groq en la barra lateral para comenzar.")
else:
    client = Groq(api_key=groq_api_key)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Entrada de Texto")
        user_input = st.text_area("Escribe una frase para analizar:", "La inteligencia artificial transforma el mundo.")
        
        if st.button("Analizar e Inferir"):
            try:
                # 1. Simulación de Tokenización (Visualización)
                st.info("### 1. Proceso de Tokenización")
                # Nota: Groq no expone el tokenizador exacto vía API, usamos una aproximación visual
                tokens = user_input.split() 
                colors = ["#FF4B4B", "#1C83E1", "#00C04A", "#FFD700", "#8A2BE2"]
                
                html_tokens = ""
                for i, token in enumerate(tokens):
                    color = colors[i % len(colors)]
                    html_tokens += f'<span style="background-color:{color}44; border:1px solid {color}; padding:2px 6px; margin:2px; border-radius:4px;">{token}</span>'
                
                st.markdown(html_tokens, unsafe_allow_html=True)
                st.caption(f"Total aproximado de tokens: {len(tokens)}")

                # 2. Inferencia con Groq
                response = client.chat.completions.create(
                    model=model_option,
                    messages=[{"role": "user", "content": user_input}],
                    temperature=temp,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                
                st.subheader("Respuesta del Modelo")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("Geometría de Embeddings (Simulación)")
        st.write("Visualización de la relación semántica en un plano 2D.")
        
        if user_input:
            # Simulamos vectores de alta dimensionalidad para los tokens
            # En un entorno real, aquí llamarías a un modelo de embeddings.
            words = user_input.split()
            if len(words) > 1:
                # Generamos vectores aleatorios pero con una semilla basada en la palabra para consistencia
                np.random.seed(42)
                vectors = np.random.randn(len(words), 64) # 64 dimensiones
                
                # Reducción de dimensionalidad con PCA a 2D
                pca = PCA(n_components=2)
                coords = pca.fit_transform(vectors)
                
                df = pd.DataFrame({
                    'Token': words,
                    'X': coords[:, 0],
                    'Y': coords[:, 1]
                })

                fig = px.scatter(df, x='X', y='Y', text='Token', 
                                 title="Mapa Semántico de Tokens (PCA)",
                                 color_discrete_sequence=['#1C83E1'])
                fig.update_traces(textposition='top center', marker=dict(size=12))
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **¿Qué estamos viendo?**
                Los modelos LLM convierten palabras en vectores numéricos. 
                Palabras con contextos similares tienden a agruparse en este espacio matemático.
                """)
            else:
                st.write("Escribe más de una palabra para ver la relación geométrica.")

# --- FOOTER ---
st.divider()
st.caption("Desarrollado para fines educativos usando Groq Cloud SDK.")
