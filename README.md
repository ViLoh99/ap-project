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
- `/results`: Stores generated results such as pickled models, word clouds, and analysis output.

## Data

The Data used for this project are the Bundesrat files from the German Parliament Corpus (GerParCor). 
- http://lrec2022.gerparcor.texttechnologylab.org
- G. Abrami, M. Bagci, L. Hammerla, and A. Mehler, “German Parliamentary Corpus (GerParCor),” in Proceedings of the Language Resources and Evaluation Conference, Marseille, France, 2022, pp. 1900-1906.
