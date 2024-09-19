import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter

# Function to load the pickle file
def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to display the top words of each topic
def display_topics(top_words_per_topic):
    for topic_id, words in top_words_per_topic.items():
        print(f"Topic {topic_id}: {', '.join(words)}")

# Function to visualize word cloud for topics
def visualize_wordclouds(top_words_per_topic):
    for topic_id, words in top_words_per_topic.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Topic {topic_id}")
        plt.show()

# Function to visualize lemma frequency for a specific document
def visualize_lemma_frequency(lemmas, doc_id):
    # Convert the list of lemmas with their counts to a dictionary
    most_common_lemmas = dict(lemmas)

    # Bar Plot for Lemma Frequency
    lemmas, counts = zip(*most_common_lemmas.items())  # Unzips into two lists for bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(lemmas, counts, color='skyblue')
    plt.title(f"Top 10 Lemmas by Frequency for Document {doc_id}")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

# Function to display the number of lemmas, topic words, intersections,
# and the percentage of topic words overlapping with lemmas
def display_summary(results):
    most_frequent_lemmas = set()
    all_topic_words = set()

    # Update logic to handle list of tuples
    for result in results['comparison_results']:
        lemmas = [lemma for lemma, count in result['most_frequent_lemmas']]  # Extract the lemmas from the list of tuples
        most_frequent_lemmas.update(lemmas)

        # Collect topic words from all dominant topics
        dominant_topic_words = set(chain.from_iterable([results['top_words_per_topic'][topic] for topic in result['dominant_topics']]))
        all_topic_words.update(dominant_topic_words)

    # Calculate intersections
    intersecting_words = most_frequent_lemmas.intersection(all_topic_words)
    total_topic_words = len(all_topic_words)
    intersection_size = len(intersecting_words)

    # Calculate percentage of topic words overlapping with lemmas
    topic_intersection_ratio = (intersection_size / total_topic_words * 100) if total_topic_words > 0 else 0

    # Display summary
    print("\nSummary:")
    print(f"Total Most Frequent Lemmas: {len(most_frequent_lemmas)}")
    print(f"Total Topic Words: {total_topic_words}")
    print(f"Intersecting Lemmas and Topic Words: {intersection_size}")
    print(f"Percentage of Topic Words Overlapping with Lemmas: {topic_intersection_ratio:.2f}%")


# Function to display Jaccard similarity results for each document and globally
def display_jaccard_similarities(results, visualize_lemmas=False):
    global_jaccard_sim = []
    for result in results['comparison_results']:
        lemmas = set([lemma for lemma, count in result['most_frequent_lemmas']])
        common_words = set(result['common_words'])
        jaccard_sim = len(common_words) / len(lemmas.union(common_words)) if len(lemmas.union(common_words)) > 0 else 0
        global_jaccard_sim.append(jaccard_sim)

        print(f"Document {result['document']}: Jaccard Similarity = {jaccard_sim:.3f}")

        # Visualize Lemma Frequency if the option is enabled
        if visualize_lemmas:
            visualize_lemma_frequency(result['most_frequent_lemmas'], result['document'])
    
    # Print global Jaccard similarity (average)
    avg_jaccard_similarity = sum(global_jaccard_sim) / len(global_jaccard_sim)
    print(f"\nGlobal Jaccard Similarity (Average for all documents): {avg_jaccard_similarity:.3f}")

# Load the pickle file and display the results
filename = '/home/ampharis/Dokumente/AP/Extracted Text - protoype code/results_decade_2.pkl'  # Change to your actual file path
results = load_results(filename)

# Display Topics and their Top Words
print("Top Words Per Topic:")
display_topics(results['top_words_per_topic'])

# Visualize Word Clouds for each topic
visualize_wordclouds(results['top_words_per_topic'])

# Display Summary Information (lemmas, topic words, intersections)
display_summary(results)

# Display Jaccard Similarity results for each document and global average, with optional lemma visualization
print("\nJaccard Similarity Results:")
display_jaccard_similarities(results, visualize_lemmas=False)  # Set to True to enable lemma frequency visualization
