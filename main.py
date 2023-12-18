# main.py
# Import library yang dibutuhkan
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.utils import custom_object_scope
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from attention import AttentionLayer

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan tokenizer yang sudah di-train
with custom_object_scope({'AttentionLayer': AttentionLayer}):
    model = load_model('text_summarization_model.h5')
    tokenizer = Tokenizer()

# Definisikan text yang akan dijadikan ringkasan
text = 'yg kenal lele legendaris lele nya crispy enak banget nasi uduk gak recommended banget nya sih pedes ya overall'

# Tokenisasi dan pad teks
tokenizer.fit_on_texts([text])
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=100)

# Melakukan prediksi menggunakan model
prediction = model.predict(padded_sequence)

# Print hasil prediksi
result = {'prediction': prediction.tolist()}
print(result)





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
