# Topic Modeling and (Lemma) Analysis

This project performs **topic modeling** on German texts using **Gensim** and **NLTK**, with a focus on comparing topics and frequent lemmas. The project also calculates **Jaccard similarity** between topics and lemmas, and visualizes the results using word clouds and frequency plots.

## Features
- **Preprocessing data**: Uses NLTK to preprocess data for Topic modeling.
- **Topic Modeling**: Uses LDA (Latent Dirichlet Allocation) to extract topics from documents.
- **Lemma Frequency Analysis**: Extracts and visualizes the most frequent lemmas in each document.
- **Jaccard Similarity**: Compares the overlap between topic words and frequent lemmas.
- **Word Clouds**: Visualizes the topics using word clouds.
-  **Comparison of Results**: Compare topic modeling results across different time spans using histograms, normal distributions, and intersection ratio visualizations.

## Directory Structure
- `/src`: Contains the core Python scripts for data processing and visualization.
  - **lda_topic_modeling.py**: Processes the documents, runs topic modeling, and extracts lemmas.
  - **read_results.py**: Loads the results, visualizes topics and lemmas, and computes Jaccard similarity.
  - **compare_results.py**: Loads and compares Jaccard similarity and topic intersection ratio across different time spans, visualizing the comparison using histograms, tables, and normal distributions.
- `/Results`: Stores Pickle files of the topic modeled data - generated with `lda_topic_modeling.py`.
- `/Comparison Results`: Stores Pickle files with Jaccard similarity and topic intersection ratios for different time spans for cross-period comparison - generated with `read_results.py`.

## Data

The Data used for this project are the Bundesrat files from the German Parliament Corpus (GerParCor). 
- http://lrec2022.gerparcor.texttechnologylab.org
- G. Abrami, M. Bagci, L. Hammerla, and A. Mehler, “German Parliamentary Corpus (GerParCor),” in Proceedings of the Language Resources and Evaluation Conference, Marseille, France, 2022, pp. 1900-1906.

## Running the Project

To get started, follow these steps to process the documents and generate the topic models:

### 1. *Process Documents*
Run the `process_documents.py` script to process your documents and generate results.
- This will traverse through all subfolders in the directory containing your .xmi files and process them using topic modeling and lemma frequency analysis.
- Results will be saved in separate pickle files for each folder.

Make sure to modify the base_directory path inside `process_documents.py` to point to your main folder containing the subfolders with the .xmi files.

### 2. *Analyze Results*
Once the documents have been processed and saved as pickle files, you can use the `read_results.py` script to visualize and analyze the results.
- This script will load the saved results and allow you to visualize the word clouds, Jaccard similarity, and other data insights.
- he Jaccard similarity and topic intersection ratio will be saved in a `Comparison Results` folder for future comparison purposes.

Make sure to modify the filepath inside `read_results.py` to point to your .pkl file

### 3. *Compare Results**: Ensure running the code leads to the expected output.
Use the `compare_results.py` script to compare Jaccard similarity and topic intersection ratio across different time spans.
- This script will load the saved comparison files from the `/Comparison Results` folder or specific provided file paths.
It visualizes Jaccard similarity and topic intersection ratio using histograms, normal distribution plots, and tables.

### 4. *Expected Results**: Ensure running the code leads to the expected output.
This section of the `README.md` ensures that users know exactly what to expect after running your code and what results they should see.

## Expected Results

After running the `process_documents.py` script, the following results will be saved:

- **Pickle Files**: For each subfolder, a pickle file will be generated that contains:
  - Preprocessed documents.
  - Topic modeling results (LDA).
  - Dominant topics per document.
  - Most frequent lemmas for each document.
  - Jaccard similarity calculations.
  - Topic intersection ratio.

Example of saved pickle files: 
- `topic_model_results_2016-2020.pkl`
- `data_topic_model_2016-2020.pkl`

### Analyzing Results

After processing, you can load and analyze the results using `read_results.py`. Expected outputs include:

1. **Word Clouds**: Word clouds for each topic, displaying the top words per topic.
2. **Jaccard Similarity**: Jaccard similarity between the most frequent lemmas and topic words, displayed per document and globally.
3. **Summary Statistics**: The script will display a summary of:
   - Total most frequent lemmas.
   - Total topic words.
   - Percentage of overlap between lemmas and topic words.
   
To visualize lemma frequency for specific documents or topics, set `visualize_lemmas=True` in `read_results.py`.

You can then use `compare_results.py` to visualize and compare Jaccard similarity and topic intersection ratio between different time spans.
