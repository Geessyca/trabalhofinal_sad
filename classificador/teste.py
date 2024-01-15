import json

import joblib

class SentimentClassifier:
    def __init__(self):
        self.loaded_model = joblib.load('classificador/modelo_naive_bayes.joblib')
        self.loaded_vectorizer = joblib.load('classificador/vetorizador.joblib')

        # Inicializar a lista de resultados
        self.results_list = []

    def classify_comments(self, comments):
        # Vetorização dos novos comentários
        comments_vectorized = self.loaded_vectorizer.transform(comments)

        # Predições dos novos comentários
        predictions = self.loaded_model.predict(comments_vectorized)

        # Adicionar novos resultados à lista
        for comment, prediction in zip(comments, predictions):
            result_dict = {"comentario": comment, "sentimento": prediction}
            self.results_list.append(result_dict)

        return self.results_list

    def save_results_to_json(self, output_file_path):
        # Salvar a lista em um arquivo JSON
        with open(output_file_path, 'a') as output_file:
            for result_dict in self.results_list:
                json.dump(result_dict, output_file)
                output_file.write('\n') # Adicionar uma nova linha para separar os resultados

        print(f'Novos resultados adicionados em {output_file_path}')
