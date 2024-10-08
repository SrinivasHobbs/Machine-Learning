from flask import Flask, request, render_template, jsonify, session
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
import io

app = Flask(__name__)
app.secret_key = "b'\xc8F\x1b\xee\x04,\xc3\x06\xe6\xa7E\xe5M$e\xba\n\xdeu\xb3\x02\xadQ\x0c"  # Replace with your actual secret key

lda_model = None
dictionary = None

def preprocess_text(text):
    if isinstance(text, str):
        return text.lower().split()
    else:
        return []

def get_topics(corpus, model, dictionary):
    topics = {}
    for idx, topic in model.show_topics(formatted=False):
        topics[f'Topic {idx}'] = [word for word, _ in topic]
    return topics

def plot_word_cloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    return base64.b64encode(img.getvalue()).decode('utf8')

@app.route('/')
def index():
    return render_template('index.html', topics={})

@app.route('/get_topics', methods=['POST'])
def get_topics_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        reviews_df = pd.read_csv(file)
        reviews_df['content'].fillna('', inplace=True)
        reviews_df['processed_content'] = reviews_df['content'].apply(preprocess_text)
        texts = reviews_df['processed_content'].tolist()

        dictionary = Dictionary(texts)
        new_corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = LdaModel(new_corpus, num_topics=10, id2word=dictionary, passes=10)
        topics = get_topics(new_corpus, lda_model, dictionary)

        # Store topics in session
        session['topics'] = topics

        return render_template('topics.html', topics=topics)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/select_topics', methods=['POST'])
def select_topics():
    accepted_topics = request.form.getlist('accepted_topics')
    topics = session.get('topics', {})

    # Generate word clouds for accepted topics
    word_clouds = {}
    for topic in accepted_topics:
        if topic in topics:
            word_clouds[topic] = plot_word_cloud(topics[topic])

    return render_template('accepted_topics.html', word_clouds=word_clouds)

if __name__ == '__main__':
    app.run(debug=True)
