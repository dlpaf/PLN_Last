import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from etiqueta import Etiquetador
from modelo_cnn import ModeloCNN
import preprocess
from tensorflow.keras.utils import to_categorical

def plotar_metricas_completas(historico, modelo, X_teste, y_teste):
    plt.figure(figsize=(18, 6))
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(historico.history['accuracy'], label='Treino')
    plt.plot(historico.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    
    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(historico.history['loss'], label='Treino')
    plt.plot(historico.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    
    # Matriz de Confusão
    plt.subplot(1, 3, 3)
    y_pred = modelo.predict(X_teste).argmax(axis=1)
    y_true = y_teste.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Verdadeira')
    
    plt.tight_layout()
    plt.savefig('metricas_completas.png')
    plt.close()

def main():
    # Caminho para o PDF combinado
    caminho_pdf = 'arquivo_combinado.pdf'
    
    # Preprocessamento
    preprocessador = preprocess.Preprocessamento(caminho_pdf)
    dados = preprocessador.carregar_dados()
    dados_processados, tokenizer = preprocessador.preparar_dados(dados)
    
    # Etiquetagem
    etiquetador = Etiquetador(dados)
    labels, label_encoder = etiquetador.codificar_labels()
    labels_categoricas = to_categorical(labels)
    
    # Divisão de dados
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados_processados, labels_categoricas, 
        test_size=0.2, random_state=42
    )
    
    # Modelo CNN
    cnn = ModeloCNN(
        input_dim=len(tokenizer.word_index) + 1, 
        output_dim=len(labels_categoricas[0]),
        max_len=dados_processados.shape[1]
    )
    modelo = cnn.construir_modelo()
    
    # Treinamento
    historico = modelo.fit(
        X_treino, y_treino,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Avaliação
    avaliacao = modelo.evaluate(X_teste, y_teste)
    print(f"Perda de Teste: {avaliacao[0]}")
    print(f"Acurácia de Teste: {avaliacao[1]}")
    
    # Plotagem
    plotar_metricas_completas(historico, modelo, X_teste, y_teste)

if __name__ == "__main__":
    main()