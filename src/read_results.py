import pickle
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter

# Function to load the results from a pickle file
def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to display the top words of each topic from the topic modelling
def display_topics(top_words_per_topic):
    for topic_id, words in top_words_per_topic.items():
        print(f"Topic {topic_id}: {', '.join(words)}")

# Function to visualize word clouds for each topic
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

# Function to display a summary of lemmas, topic words, intersections, and the percentage of topic words overlapping with lemmas
def display_summary(results):
    most_frequent_lemmas = set()
    all_topic_words = set()

    # Collect lemmas and topic words from each document
    for result in results['comparison_results']:
        lemmas = [lemma for lemma, count in result['most_frequent_lemmas']]  # Extract the lemmas from the list of tuples
        most_frequent_lemmas.update(lemmas)

        # Collect topic words from all dominant topics
        dominant_topic_words = set(chain.from_iterable([results['top_words_per_topic'][topic] for topic in result['dominant_topics']]))
        all_topic_words.update(dominant_topic_words)

    # Calculate intersections between lemmas and topic words
    intersecting_words = most_frequent_lemmas.intersection(all_topic_words)
    total_topic_words = len(all_topic_words)
    intersection_size = len(intersecting_words)

    # Calculate percentage of topic words overlapping with lemmas
    topic_intersection_ratio = (intersection_size / total_topic_words * 100) if total_topic_words > 0 else 0

    # Count the number of processed documents
    num_docs = len(results['comparison_results'])

    # Display a summary with information
    print("\nSummary:")
    print(f"Number of Processed Documents: {num_docs}")
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
    
    # Print global Jaccard similarity (average accross all documents)
    avg_jaccard_similarity = sum(global_jaccard_sim) / len(global_jaccard_sim)
    print(f"\nGlobal Jaccard Similarity (Average for all documents): {avg_jaccard_similarity:.3f}")

# Function to save Jaccard similarity results and topic_intersection_ratio
def save_jaccard_and_topic_intersection(results, input_filename):
    jaccard_data = []
    topic_intersection_ratio = 0  # Initialize ratio for calculation

    # Calculate Jaccard similarities for each document
    for result in results['comparison_results']:
        lemmas = set([lemma for lemma, count in result['most_frequent_lemmas']])
        common_words = set(result['common_words'])
        jaccard_sim = len(common_words) / len(lemmas.union(common_words)) if len(lemmas.union(common_words)) > 0 else 0
        jaccard_data.append(jaccard_sim)

    # Calculate topic intersection ratio
    most_frequent_lemmas = set()
    all_topic_words = set()

    for result in results['comparison_results']:
        lemmas = [lemma for lemma, count in result['most_frequent_lemmas']]  # Extract the lemmas from the list of tuples
        most_frequent_lemmas.update(lemmas)

        # Collect topic words from all dominant topics
        dominant_topic_words = set(chain.from_iterable([results['top_words_per_topic'][topic] for topic in result['dominant_topics']]))
        all_topic_words.update(dominant_topic_words)

    intersecting_words = most_frequent_lemmas.intersection(all_topic_words)
    total_topic_words = len(all_topic_words)
    intersection_size = len(intersecting_words)
    
    if total_topic_words > 0:
        topic_intersection_ratio = (intersection_size / total_topic_words) * 100  # Calculate percentage

    # Extract time span from input filename
    time_span = os.path.basename(input_filename).replace('results_', '').replace('.pkl', '')

    # Create 'Comparison Results' folder if it doesn't exist
    comparison_folder = 'Comparison Results'
    if not os.path.exists(comparison_folder):
        os.makedirs(comparison_folder)

    save_filename = os.path.join(comparison_folder, f'data_{time_span}.pkl')

    # Save both Jaccard similarity and topic_intersection_ratio to file
    save_data = {
        'jaccard_similarities': jaccard_data,
        'topic_intersection_ratio': topic_intersection_ratio
    }

    with open(save_filename, 'wb') as f:
        pickle.dump(save_data, f)


## Main execution point

# Load the results from a pickle file
filename = '/home/ampharis/Dokumente/AP/Project/Results/topic_model_results_2021-2025.pkl'  # Change to actual file path
results = load_results(filename)

# Display the top words for each topic
print("Top Words Per Topic:")
display_topics(results['top_words_per_topic'])

# Visualize word clouds for each topic
visualize_wordclouds(results['top_words_per_topic'])

# Display the summary of lemmas, topic words, intersections, and overlap percentage
display_summary(results)

# Display Jaccard similarity results for each document and global average
print("\nJaccard Similarity Results:")
display_jaccard_similarities(results, visualize_lemmas=False)  # Set to True to enable lemma frequency visualization

# Save the Jaccard similarity results and topic_intersection_ratio
save_jaccard_and_topic_intersection(results, filename)
