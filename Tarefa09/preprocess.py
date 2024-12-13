import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googletrans import Translator

class Preprocessamento:
    def __init__(self, caminho_pdf):
        self.caminho_pdf = caminho_pdf
        self.translator = Translator()

    def carregar_dados(self):
        import PyPDF2

        dados = []
        with open(self.caminho_pdf, 'rb') as arquivo:
            leitor_pdf = PyPDF2.PdfReader(arquivo)
            for pagina in leitor_pdf.pages:
                texto = pagina.extract_text()
                dados.append(texto)

        return dados

    def limpar_texto(self, texto):
        import re
        # Remover URLs, emails e outros padrões
        texto = re.sub(r'http\S+|www\S+|\S+@\S+\.\S+', '', texto)
        # Remover caracteres especiais e números
        texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚâêîôûÂÊÎÔÛãõÃÕçÇ\s]', '', texto)
        # Remover espaços extras
        texto = re.sub(r'\s+', ' ', texto).strip()
        # Converter para minúsculas
        texto = texto.lower()
        return texto

    def traduzir_texto(self, texto):
        try:
            # Traduzir texto de inglês para português
            texto_traduzido = self.translator.translate(texto, src='en', dest='pt').text
            return texto_traduzido
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return texto

    def preparar_dados(self, dados, max_palavras=5000, max_len=200):
        # Limpar e traduzir textos
        dados_processados = []
        for texto in dados:
            texto_limpo = self.limpar_texto(texto)
            texto_traduzido = self.traduzir_texto(texto_limpo)
            dados_processados.append(texto_traduzido)

        # Tokenização
        tokenizer = Tokenizer(num_words=max_palavras)
        tokenizer.fit_on_texts(dados_processados)

        # Conversão para sequências
        sequencias = tokenizer.texts_to_sequences(dados_processados)

        # Padding
        dados_padded = pad_sequences(sequencias, maxlen=max_len)

        return dados_padded, tokenizer
