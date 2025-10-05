# 🚀 Análise de Sentimentos em Redes Sociais - Campus Party Weekend Piauí 2025

Este repositório contém o código e os recursos para o projeto de extensão "Análise de Sentimentos em Redes Sociais utilizando Machine Learning", desenvolvido para apresentação na **Campus Party Weekend Piauí 2025**.

O projeto é uma iniciativa do curso de **Engenharia de Computação com IA** do Centro Universitário Tecnológico de Teresina (UNI-CET).

## 🎯 Objetivo do Projeto

O objetivo geral é desenvolver um projeto que utilize técnicas de Machine Learning para analisar sentimentos em textos de redes sociais, proporcionando conhecimento tecnológico e uma devolutiva social sobre a percepção digital em temas locais.

### Objetivos Específicos:
* Capacitar os estudantes na aplicação prática de técnicas de Processamento de Linguagem Natural (PLN).
* Coletar e processar dados textuais de redes sociais públicas.
* Treinar e avaliar modelos de análise de sentimentos para o português.
* Desenvolver um painel interativo de visualização de dados para apresentar os resultados de forma clara e acessível.

## 👥 Equipe

### Discentes
* Rafael Oliveira
* Ailton Medeiros
* Lais Eulálio
* Antônio Wilker
* Isaac Aragão
* Paula Iranda

### Docente Orientador
* Prof. Dr. Artur Felipe da Silva Veloso

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Desenvolvimento de ML:** Google Colab
* **Bibliotecas de Dados:** Pandas, NLTK
* **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`)
* **Dashboard Interativo:** Streamlit
* **Versionamento:** Git & GitHub

## 📁 Estrutura do Repositório

```
.
├── data/
│   ├── dataset_bruto.csv       # Os dados de comentários do Reddit usados no treinamento.
│   └── dados_limpos.csv        # Dados após limpeza e pré-processamento (gerado pelo notebook).
├── saved_models/
│   └── modelo_sentimento.joblib # O pipeline de ML treinado (TF-IDF + LR).
├── notebooks/
│   └── Desenvolvimento_Analise_Sentimentos.ipynb # Notebook com todo o processo de ML.
├── app.py                      # Script principal da aplicação com Streamlit.
├── requirements.txt            # Lista de dependências exatas do projeto.
└── README.md                   # Este arquivo.
```

## ▶️ Como Executar o Projeto Localmente

Para executar o painel interativo na sua máquina, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/analise-sentimentos-cpweekend.git](https://github.com/SEU-USUARIO/analise-sentimentos-cpweekend.git)
    cd analise-sentimentos-cpweekend
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Crie o ambiente
    python -m venv venv

    # Ative o ambiente (Windows)
    .\venv\Scripts\activate

    # Ative o ambiente (Linux/Mac)
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação com Streamlit:**
    ```bash
    streamlit run app.py
    ```
    Seu navegador abrirá automaticamente com o dashboard interativo.

## 📊 Status do Projeto

**COMPLETO** e pronto para apresentação na Campus Party Weekend Piauí 2025.
