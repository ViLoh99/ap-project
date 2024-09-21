import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
from scipy.stats import norm
import pandas as pd

# Load Jaccard similarity data and topic_intersection_ratio from files
def load_data(filenames):
    jaccard_data = []
    topic_intersection_ratios = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            jaccard_data.append(data['jaccard_similarities'])  # Load Jaccard similarities
            topic_intersection_ratios.append(data.get('topic_intersection_ratio', 0))  # Load intersection ratio, default to 0 if not found
    return jaccard_data, topic_intersection_ratios


# Function to display histograms back to back
def plot_histograms(jaccard_data_list, labels):
    plt.figure(figsize=(10, 6))
    for data, label in zip(jaccard_data_list, labels):
        plt.hist(data, alpha=0.7, label=label, bins=15, edgecolor='black')  # Adjusted for better visibility
    
    plt.title("Jaccard Similarity Histograms")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Function to calculate and display mean and standard deviation
def calculate_statistics(jaccard_data_list, labels):
    means = []
    std_devs = []
    for data, label in zip(jaccard_data_list, labels):
        mean = np.mean(data)
        std_dev = np.std(data)
        print(f"{label}: Mean = {mean:.3f}, Standard Deviation = {std_dev:.3f}")
        means.append(mean)
        std_devs.append(std_dev)
    return means, std_devs

# Function to toggle visibility of a line in the plot
def toggle_visibility(label, lines, labels):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())  # Toggle the visibility of the corresponding line
    plt.draw()  # Redraw the plot to reflect the changes

# Interactive normal distribution plot with legend on the right and checkboxes on the left
def plot_normal_distribution_interactive(jaccard_data_list, labels):
    # Create the main plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.3, right=0.8)  # Adjust the space to fit both checkboxes on the left and legend on the right

    lines = []
    valid_labels = []  # To store labels with visible lines
    # Create the normal distribution lines for each dataset
    for data, label in zip(jaccard_data_list, labels):
        mean = np.mean(data)
        std_dev = np.std(data)

        # Check if standard deviation is zero
        if std_dev == 0:
            print(f"Skipping dataset '{label}' due to zero standard deviation.")
            plt.plot(mean, 0, marker='o', markersize=5, label=f"{label} (zero std dev)")
            continue

        x = np.linspace(min(data), max(data), 100)
        y = norm.pdf(x, mean, std_dev)

        # Only plot if data is valid
        line, = ax.plot(x, y, label=label)
        lines.append(line)
        valid_labels.append(label)  # Add the label only if a valid line is created

    ax.set_title("Normal Distribution of Jaccard Similarity")
    ax.set_xlabel("Jaccard Similarity")
    ax.set_ylabel("Density")

    # Place the legend outside the plot on the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the checkboxes to match only the valid labels
    check_ax = plt.axes([0.01, 0.4, 0.2, 0.5])  # Position for the checkboxes on the left of the plot
    check = CheckButtons(check_ax, valid_labels, [True] * len(valid_labels))  # Create checkboxes for valid lines

    # Register callback function to toggle visibility
    check.on_clicked(lambda label: toggle_visibility(label, lines, valid_labels))

    plt.show()


# Function to extract only the time span from a filename (e.g., '2021-2025')
def extract_time_span(filename):
    # Remove 'jaccard_data_' and '.pkl' to extract just the time span
    return os.path.basename(filename).replace('data_', '').replace('topic_model_', '').replace('.pkl', '')

# Function to load and compare specific files or all files in the "Comparison Results" folder
def compare_results(files=None, comparison_folder='Comparison Results'):
    # If no specific files are provided, use all files in the folder
    if files is None:
        files = [os.path.join(comparison_folder, f) for f in os.listdir(comparison_folder) if f.endswith('.pkl')]
    
    if len(files) == 0:
        print("No Jaccard data files found.")
        return

    # Extract labels (time spans) from file names
    labels = [extract_time_span(file) for file in files]
    
    # Load Jaccard similarity data and intersection ratios
    jaccard_data_list, intersection_ratios = load_data(files)
    
    # Plot histograms if there are 2 or fewer files
    if len(files) <= 2:
        plot_histograms(jaccard_data_list, labels)
    else:
        print("More than two files, skipping histogram.")
    
    # Calculate statistics and plot mean/std deviation graph
    means, std_devs = calculate_statistics(jaccard_data_list, labels)
    plot_mean_std_graph(means, std_devs, labels)

    # Plot normal distributions with interactive toggle for visibility
    plot_normal_distribution_interactive(jaccard_data_list, labels)
    
    # Display topic intersection ratios in table format
    display_intersection_ratios_table(intersection_ratios, labels)

# Function to display the intersection ratios in a table
def display_intersection_ratios_table(ratios, labels):
    fig, ax = plt.subplots(figsize=(6, 4))  # Set the size of the table

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Define table data and column labels
    table_data = list(zip(labels, [f"{ratio:.2f}%" for ratio in ratios]))
    column_labels = ["Time Span", "Topic Intersection Ratio (%)"]

    # Create and display the table
    table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')

    # Adjust font size and table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # Adjust scaling for better readability

    # Add the title above the table
    plt.suptitle("Topic Intersection Ratios", fontsize=14, y=1.15)

    plt.show()

# Function to plot the mean and standard deviation graph
def plot_mean_std_graph(means, std_devs, labels):
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, capthick=2, color='b', ecolor='r')
    
    ax.set_title("Mean and Standard Deviation of Jaccard Similarity")
    ax.set_xlabel("Time Span")
    ax.set_ylabel("Mean Jaccard Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

## Main execution point

comparison_folder = 'Comparison Results'  # Folder where all the Jaccard data files are stored

# For comparing all results in the 'Comparison Results' folder
compare_results()

# For comparing all specific files
specific_files = ['path/file_1.pkl', 'path/file_2.pkl']    # Update with actual file path
#compare_results(files=specific_files)
