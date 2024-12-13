from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

class ModeloCNN:
    def __init__(self, input_dim, output_dim, max_len):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
    
    def construir_modelo(self):
        """
        Constrói uma CNN com camadas de Dropout e BatchNormalization
        """
        modelo = Sequential([
            # Camada de Embedding
            Embedding(self.input_dim, 128, input_length=self.max_len),
            
            # Primeira camada convolucional
            Conv1D(64, 5, activation='relu'),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            
            # Camadas densas com Dropout
            Dense(64, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            # Camada de saída
            Dense(self.output_dim, activation='softmax')
        ])
        
        modelo.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return modelo
