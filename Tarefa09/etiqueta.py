from sklearn.preprocessing import LabelEncoder

class Etiquetador:
    def __init__(self, dados):
        self.dados = dados
        self.label_encoder = LabelEncoder()
    
    def codificar_labels(self):
        """
        Codifica as labels para formato numérico
        """
        labels_numericas = self.label_encoder.fit_transform(self.dados)
        return labels_numericas, self.label_encoder

