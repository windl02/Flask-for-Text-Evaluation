from flask import Flask, render_template, jsonify, request
import torch

from pred import predict_sentiment
from vocabulary import vocab
from modelclass import model

device = torch.device('cpu')
model.load_state_dict(torch.load('model.pt', map_location=device))

app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         sentence = request.form['text-input']
#         res = predict_sentiment(model, sentence, vocab, device)
#         print(res)
#         return render_template('index.html', input=res)
    
#     return render_template('index.html')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['text-input']
        res = predict_sentiment(model, sentence, vocab, device)
        return jsonify({'prediction': res})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
