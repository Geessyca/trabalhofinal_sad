import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Adicionei a importação para suporte CORS
import sys
sys.path.append(r'.')
from classificador.teste import SentimentClassifier

app = Flask(__name__)
CORS(app)  # Adicionei essa linha para habilitar o suporte a CORS

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo vazio'})

    # Processar o conteúdo do arquivo e criar uma lista
    file_content = file.read().decode('utf-8')
    file_lines = file_content.split('\n')
    
    # Remover linhas em branco
    file_lines = [line.strip() for line in file_lines if line.strip()]

    # Chamar a função específica com a lista e retornar o resultado
    result = classificar(file_lines)

    return jsonify({'result': result})

def classificar(lista):
    classifier = SentimentClassifier()
    classifier.classify_comments(lista)
    output_file_path = 'resultados.json'
    classifier.save_results_to_json(output_file_path)
    return "Classificado"

def ler_arquivo(nome_arquivo):
    with open(nome_arquivo, 'r') as file:
        linhas = file.readlines()
        lista_json = [json.loads(linha) for linha in linhas]
    return lista_json

@app.route('/obter_lista_json', methods=['GET'])
def obter_lista_json():
    nome_arquivo = 'resultados.json'  # Substitua pelo nome do seu arquivo
    lista_json = ler_arquivo(nome_arquivo)
    return jsonify(lista_json)

@app.route('/remove_list', methods=['GET'])
def remove_list():
    nome_arquivo = 'resultados.json'  # Substitua pelo nome do seu arquivo
    with open(nome_arquivo, 'w') as file:
        file.write("")
    return jsonify({'result': "Removidos"})


if __name__ == '__main__':
    app.run(debug=True)
