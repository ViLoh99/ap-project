import gensim
from gensim import corpora, models
from itertools import chain
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as ET
import os
from collections import Counter
import pickle
import string
import re

# Function to parse the XML file and prepare raw text
def extract_raw_text(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract the raw text
    raw_text = root.find('.//cas:Sofa', namespaces={'cas': 'http:///uima/cas.ecore'}).get('sofaString')
    
    return prepare_text(raw_text)

# Function to prepare raw text for topic modeling
def prepare_text(raw_text):
    normalized_text = raw_text.lower()
    normalized_text = normalized_text.translate(str.maketrans("","",string.punctuation))
    normalized_text = re.sub(r"[^a-zA-Züäöß\s]", "", normalized_text)
    tokens = word_tokenize(normalized_text)

    # POS Tagging and Filtering
    pos_tags = nltk.pos_tag(tokens, tagset='universal')
    pos_tags_to_keep = {"NOUN", "VERB", "ADJ"}
    filtered_tokens = [word for word, pos in pos_tags if pos in pos_tags_to_keep]
    
    stop_words = set(stopwords.words('german'))
    filtered_text = [word for word in filtered_tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    
    return lemmatized_text

# Function to process documents in a folder and apply topic modeling, frequency analysis, etc.
def process_documents(folder_path, topn=25):
    prep_docs = []
    
    # Walk through all .xmi files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xmi'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                prep_docs.append(extract_raw_text(file_path))

    # Topic Modeling
    dictionary = corpora.Dictionary(prep_docs)
    dictionary.filter_extremes(no_below=1, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in prep_docs]
    
    # Apply TF-IDF and LDA Model
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda_model = models.LdaModel(corpus_tfidf, num_topics=15, id2word=dictionary, passes=20, eta=0.15)

    # Extract topic words
    top_words_per_topic = {}
    for topic_id in range(lda_model.num_topics):
        top_words = lda_model.show_topic(topic_id, topn=topn)
        top_words_per_topic[topic_id] = [word for word, prob in top_words]
    
    # Dominant topics per document
    dominant_topics_per_doc = []
    dominance_threshold = 0.2
    for doc_id, doc_topics in enumerate(lda_model[corpus]):
        sorted_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        dominant_topics = [topic[0] for topic in sorted_topics if topic[1] >= dominance_threshold]
        if not dominant_topics:
            dominant_topics = [sorted_topics[0][0]]
        dominant_topics_per_doc.append(dominant_topics)

    # Most frequent lemmas and comparison
    flattened_lemmas = list(chain.from_iterable(prep_docs))
    lemma_counter = Counter(flattened_lemmas)
    frequency_threshold = 0.5 * len(prep_docs)
    common_lemmas = {lemma for lemma, count in lemma_counter.items() if count > frequency_threshold}
    
    filtered_docs = [[lemma for lemma in doc if lemma not in common_lemmas] for doc in prep_docs]
    
    # Collect results for each document
    comparison_results = []
    for doc_id, lemmas in enumerate(filtered_docs):
        dominant_topics = dominant_topics_per_doc[doc_id]
        common_words = set()
        for dominant_topic in dominant_topics:
            topic_words = top_words_per_topic[dominant_topic]
            common_words.update(set(lemmas).intersection(set(topic_words)))
        comparison_results.append({
            'document': doc_id,
            'dominant_topics': dominant_topics,
            'most_frequent_lemmas': Counter(lemmas).most_common(10),
            'common_words': list(common_words)
        })

    # Return the results for external saving or further processing
    return {
        'prep_docs': prep_docs,
        'top_words_per_topic': top_words_per_topic,
        'dominant_topics_per_doc': dominant_topics_per_doc,
        'comparison_results': comparison_results,
        'lda_model': lda_model
    }

# Function to save results using pickle
def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

# Example usage:
folder_path = '/home/ampharis/Dokumente/AP/Bundesrat/xmi/2016-2020'  # Update with actual folder path
results = process_documents(folder_path, topn=25)  # Process the folder
save_results(results, 'results_decade_2.pkl')  # Save the results to a pickle file
