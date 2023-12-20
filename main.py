# main.py
# Import library yang dibutuhkan
from flask import Flask, request, jsonify
from summarizer import preprocess_text, calculate_word_frequency, calculate_sentence_scores, calculate_similarity_matrix, sentence_similarity, apply_text_rank, generate_summary

# Inisialisasi Flask
app = Flask(__name__)

# Define the summarization route
@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        input_text = request.form['text']
        
        sentences = preprocess_text(input_text)

        word_freq = calculate_word_frequency(sentences)
        sentence_scores_freq = calculate_sentence_scores(word_freq, sentences)

        similarity_matrix = calculate_similarity_matrix(sentences)
        scores_text_rank = apply_text_rank(similarity_matrix)

        combined_scores = [0.5 * scores_text_rank[i] + 0.5 * sentence_scores_freq[i] for i in range(len(sentences))]

        N = 3  # jumlah kalimat
        final_summary = generate_summary(sentences, combined_scores, N)

        return jsonify({
            'text': input_text,
            'summary': final_summary
        })

if __name__ == '__main__':
    app.run(debug=True)
