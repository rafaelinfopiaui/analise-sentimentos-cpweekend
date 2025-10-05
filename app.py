import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import nltk

# 1. Configuração do Streamlit (DEVE SER O PRIMEIRO COMANDO ST)
st.set_page_config(page_title="Análise de Sentimentos CPWPI", layout="centered")

# --- 2. Configurações Iniciais e Carregamento do Modelo ---
MODEL_PATH = 'saved_models/modelo_sentimento.joblib' 

# Garante que 'punkt' e 'stopwords' estão disponíveis
try:
    stopwords_pt = stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    stopwords_pt = stopwords.words('portuguese')

# Carregamento do Modelo
@st.cache_resource # Carrega o modelo apenas uma vez para performance
def load_model(path):
    try:
        pipeline = joblib.load(path)
        return pipeline
    except FileNotFoundError:
        return None

pipeline = load_model(MODEL_PATH)

# --- 3. Funções de Pré-processamento e Predição ---

# Função de Pré-processamento (IDÊNTICA AO NOTEBOOK)
def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    texto = texto.lower()
    # Remover URLs
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    # Remover menções (@) e hashtags (#)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    # Remover caracteres não-alfabéticos (mantém letras e espaços)
    texto = re.sub(r'[^a-z\s]', '', texto)
    
    tokens = nltk.word_tokenize(texto)
    # Remover stopwords
    tokens_limpos = [palavra for palavra in tokens if palavra not in stopwords_pt]
    
    return ' '.join(tokens_limpos)

# Mapeamento de Sentimento (AJUSTE CONFORME SEU MODELO)
SENTIMENT_MAP = {
    '0': 'Negativo',
    '1': 'Positivo',
    'negativo': 'Negativo',
    'positivo': 'Positivo',
}

# Função de Predição
def predict_sentiment(text):
    if pipeline is None:
        return "MODEL_ERROR"
        
    processed_text = limpar_texto(text)
    
    prediction = pipeline.predict([processed_text])[0]
    
    return SENTIMENT_MAP.get(str(prediction).lower(), str(prediction))

# --- 4. Interface Principal do Streamlit ---

st.title("🧠 Classificador de Sentimentos em Português")
st.subheader("Projeto de Extensão: CP Weekend Piauí 2025")


# Verifica e exibe o status do modelo (APÓS st.set_page_config e st.title)
if pipeline is None:
    st.error(f"Erro: O arquivo do modelo (`modelo_sentimento.joblib`) não foi encontrado em `{MODEL_PATH}`.")
    st.warning("Verifique se o arquivo foi baixado do Drive e está na pasta `saved_models/`.")
    st.stop()
else:
    st.sidebar.success("Modelo de ML carregado com sucesso!")


st.markdown("""
Esta aplicação utiliza o `Pipeline` de Machine Learning treinado para classificar o sentimento.
""")

text_input = st.text_area("Insira um texto para análise:", "A Campus Party Weekend Piauí é um evento incrível, mal posso esperar!", height=150)

# Botão de Predição
if st.button("Analisar Sentimento"):
    if text_input:
        with st.spinner('Analisando o texto...'):
            sentiment = predict_sentiment(text_input)
            
            st.markdown("---")
            st.write("### Resultado da Análise:")
            
            # Formatando o output
            if sentiment == "Positivo":
                st.success(f"O sentimento é: **{sentiment}** 🎉")
            elif sentiment == "Negativo":
                st.error(f"O sentimento é: **{sentiment}** 🙁")
            elif sentiment == "MODEL_ERROR":
                st.error("Erro no Pipeline. Verifique o console para detalhes.")
            else:
                st.info(f"O sentimento é: **{sentiment}**")
            
            st.markdown("---")
    else:
        st.warning("Por favor, insira um texto para realizar a análise.")

# Rodapé/Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Desenvolvimento:** Equipe de Extensão UNI-CET")