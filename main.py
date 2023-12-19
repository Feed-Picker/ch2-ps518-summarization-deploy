# main.py
# Import library yang dibutuhkan
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.utils import custom_object_scope
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from attention import AttentionLayer
import pickle
import numpy as np
import re

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan tokenizer yang sudah di-train
model = load_model('text-sum.h5', custom_objects={'AttentionLayer': AttentionLayer})
with open('text_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_text_len=250
max_summary_len=100

# Text preprocessing functions
def preprocess_text(text):
    text = text.lower() # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text) # numbers, special characters
    text = re.sub(r'\s+', ' ', text).strip() # extra spasi
    text = re.sub(r'\\x..',' ',text) # emoji
    return text

def greedy_decode_sequence(input_seq):
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = model.predict([input_seq, target_seq])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tokenizer.index_word[sampled_token_index]

        if sampled_token != '<end>':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word
        if sampled_token == '<end>' or len(decoded_sentence.split()) >= max_summary_len:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        h, c = [h, c]

    return decoded_sentence

# Text summarization function
def summarize_text(input_text):
    input_text = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_text_len, padding='post')
    
    # Generate summary
    summary_seq = greedy_decode_sequence(input_seq)
    
    # Convert the summary back to text
    summary_text = tokenizer.sequences_to_texts([summary_seq])[0]
    
    return summary_text


# Define the summarization route
@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        input_text = request.form['text']
        summary = summarize_text(input_text)
        return jsonify({
            'text': input_text,
            'summary': summary
        })

if __name__ == '__main__':
    app.run(debug=True)

# Definisikan text yang akan dijadikan ringkasan
#text = 'yg kenal lele legendaris lele nya crispy enak banget nasi uduk gak recommended banget nya sih pedes ya overall'

# Tokenisasi dan pad teks
#tokenizer.fit_on_texts([text])
#sequence = tokenizer.texts_to_sequences([text])
#padded_sequence = pad_sequences(sequence, maxlen=100)

# Melakukan prediksi menggunakan model
#prediction = model.predict(padded_sequence)

# Print hasil prediksi
#result = {'prediction': prediction.tolist()}
#print(result)





# # Definisikan fungsi untuk melakukan prediksi
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Menerima data JSON dari permintaan
#     data = request.json
#     text = data['text']

#     # Tokenisasi dan pad teks menggunakan tokenizer yang sama
#     sequence = tokenizer.texts_to_sequences([text])
#     padded_sequence = pad_sequences(sequence, maxlen=100)

#     # Lakukan prediksi menggunakan model
#     prediction = model.predict(padded_sequence)

#     # Format hasil prediksi
#     result = {'prediction': prediction.tolist()}

#     # Mengembalikan hasil prediksi dalam bentuk JSON
#     return jsonify(result)

# # Menjalankan aplikasi Flask
# if __name__ == '__main__':
#     app.run(debug=True)
