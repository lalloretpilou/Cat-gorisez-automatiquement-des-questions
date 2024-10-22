from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import util, SentenceTransformer

# Charger le modèle
try:
	model = pickle.load(open('model', 'rb'))
	mlb = pickle.load(open('mlb_embeddings', 'rb'))
	words_embeddings = pickle.load(open('words_embeddings', 'rb'))
	all_tags = pickle.load(open('all_tags', 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, this is the prediction API!"

# Route pour prédire les tags à partir du texte
@app.route('/predict', methods=["GET", "POST"])
  # Changer GET à POST
def predict():
    # Récupérer les données envoyées sous forme de JSON
    print('before data')
    data = request.get_json()  # Utiliser get_json pour gérer le type de contenu
    print('after data')

    if data is None or 'text' not in data:
        return jsonify({"error": "No text field provided"}), 400

    # Récupérer le texte envoyé pour la prédiction
    text = data['text']

    try:
        # Faire la prédiction en utilisant le modèle chargé
        embeddings = model.encode(text)
        cosine_scores = util.cos_sim(embeddings, words_embeddings)
        predictions = [all_tags[int(e)] for e in cosine_scores.sort(descending=True)[1][0][0:3]]

        # Renvoyer la prédiction sous forme de JSON
        return jsonify({'prediction': predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Gérer les erreurs de prédiction

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Écouter sur toutes les interfaces réseau
