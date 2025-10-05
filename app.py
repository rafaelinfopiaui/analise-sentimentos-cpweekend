import streamlit as st
import pickle # AGORA USAMOS PICKLE
import re
import numpy as np # Necessário para o bypass do pickle
from nltk.corpus import stopwords
import nltk

# 1. Configuração do Streamlit (DEVE SER O PRIMEIRO COMANDO ST)
st.set_page_config(page_title="Análise de Sentimentos CPWPI", layout="centered")

# --- 2. Configurações Iniciais e Carregamento dos Componentes ---

# Caminhos para os arquivos separados, AGORA .pkl
TFIDF_PATH = 'saved_models/tfidf_vectorizer.pkl'
CLF_PATH = 'saved_models/logistic_model.pkl'

# Garante que 'stopwords' esteja acessível
try:
    stopwords_pt = stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
    stopwords_pt = stopwords.words('portuguese')


# Função para manipular a falha de compatibilidade do numpy/pickle
class CustomUnpickler(pickle.Unpickler):
    """
    Solução para o erro ModuleNotFoundError: No module named 'numpy_core'
    e outras falhas de serialização entre Colab e WSL.
    Força o carregamento das classes do numpy do módulo base.
    """
    def find_class(self, module, name):
        if module == 'numpy.core.multiarray':
            return np.__dict__[name]
        if module == 'numpy.core.umath':
            return np.__dict__[name]
        if module == 'sklearn.feature_extraction.text':
            import sklearn.feature_extraction.text
            return sklearn.feature_extraction.text.__dict__[name]
        return super().find_class(module, name)

# Função para carregar os dois componentes (Vetorizador e Classificador)
@st.cache_resource 
def load_components():
    try:
        # Carrega os dois componentes usando PICKLE com o bypass
        with open(TFIDF_PATH, 'rb') as f:
            # Usa o CustomUnpickler para carregar o vetorizador
            vectorizer = CustomUnpickler(f).load() 
        
        with open(CLF_PATH, 'rb') as f:
            # Usa o CustomUnpickler para carregar o classificador
            classifier = CustomUnpickler(f).load() 
        
        return vectorizer, classifier
    except FileNotFoundError:
        return None, None
    except Exception as e:
        # Se falhar mesmo com o bypass, imprime o erro final e retorna None
        print(f"Erro Crítico de Carregamento com Bypass: {e}")
        return None, None

vectorizer, classifier = load_components()

# --- 3. Função de Pré-processamento e Predição ---

# Função de Pré-processamento (Final e Robusta - Sincronizada com o Notebook)
def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    texto = texto.lower()
    # Remove URLs, menções, hashtags
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto) 
    texto = re.sub(r'@\w+|#\w+', '', texto)             
    
    # Remove pontuação (CRUCIAL para o .split() funcionar como tokenizador)
    texto = re.sub(r'[^\w\s]', '', texto) 
    
    # Tokenização Simples (BYPASS DO ERRO 'punkt')
    tokens = texto.split()
    
    # Remover stopwords
    stopwords_pt = stopwords.words('portuguese')
    tokens_limpos = [palavra for palavra in tokens if palavra not in stopwords_pt]
    
    return ' '.join(tokens_limpos)

# Mapeamento de Sentimento
SENTIMENT_MAP = {
    '0': 'Negativo',
    '1': 'Positivo',
    'negativo': 'Negativo',
    'positivo': 'Positivo',
}

# Função de Predição (Lógica de Componentes Separados)
def predict_sentiment(text):
    if vectorizer is None or classifier is None:
        return "MODEL_ERROR"
        
    processed_text = limpar_texto(text)
    
    # 2. Vetorização: Transforma o texto limpo usando o vetorizador
    X_vectorized = vectorizer.transform([processed_text])
    
    # 3. Classificação: Aplica o classificador no vetor
    prediction = classifier.predict(X_vectorized)[0]
    
    # Mapeia o resultado
    return SENTIMENT_MAP.get(str(prediction).lower(), str(prediction))

# --- 4. Interface Principal do Streamlit ---

st.title("🧠 Classificador de Sentimentos em Português")
st.subheader("Projeto de Extensão: CP Weekend Piauí 2025")


# Verifica e exibe o status do modelo (APÓS st.set_page_config e st.title)
if vectorizer is None or classifier is None:
    st.error(f"Erro: Arquivos do modelo não encontrados. Verifique se {TFIDF_PATH} e {CLF_PATH} estão na pasta 'saved_models/' e têm a extensão .pkl.")
    st.warning("É necessário fazer a última execução do Notebook para gerar os dois arquivos **.pkl**.")
    st.stop()
else:
    st.sidebar.success("Modelo de ML carregado com sucesso!")


st.markdown("Esta aplicação utiliza o **Pipeline** de Machine Learning treinado para classificar o sentimento.")

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
st.sidebar.markdown("### Apoio Institucional")

# Inclusão das Imagens na Sidebar
st.sidebar.image("uploaded:unicet_white.png-ad624093-8365-4be1-82c3-885eca156d1d", use_column_width=True)
st.sidebar.image("uploaded:ENG-CIA logo.jpg-0f996b2d-fba4-498c-9ebf-828b42ca9be2", use_column_width=True)
st.sidebar.image("uploaded:CPWeekend_Piaui.png-2319df34-36a1-43df-ac89-62d347622253", use_column_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Desenvolvimento:** Equipe de Extensão UNI-CET")