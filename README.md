# Topic Modeling and Lemma Analysis

This project performs **topic modeling** on German texts using **Gensim** and **NLTK**, with a focus on comparing topics and frequent lemmas. The project also calculates **Jaccard similarity** between topics and lemmas, and visualizes the results using word clouds and frequency plots.

## Features
- **Topic Modeling**: Uses LDA (Latent Dirichlet Allocation) to extract topics from documents.
- **Lemma Frequency Analysis**: Extracts and visualizes the most frequent lemmas in each document.
- **Jaccard Similarity**: Compares the overlap between topic words and frequent lemmas.
- **Word Clouds**: Visualizes the topics using word clouds.

## Directory Structure
- `/src`: Contains the core Python scripts for data processing and visualization.
  - **process_documents.py**: Processes the documents, runs topic modeling, and extracts lemmas.
  - **read_results.py**: Loads the results, visualizes topics and lemmas, and computes Jaccard similarity.
- `/data`: Raw XML files or other data sources (not included here; see instructions below for data usage).
- `/results`: Stores generated results such as pickled models, word clouds, and analysis output.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/project-name.git
   cd project-name
