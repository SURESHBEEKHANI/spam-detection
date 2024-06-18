from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return jsonify({'prediction': 'spam' if prediction[0] == 1 else 'ham'})

if __name__ == '__main__':
    app.run(debug=True)
