import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
from scipy.stats import norm

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
    for data, label in zip(jaccard_data_list, labels):
        mean = np.mean(data)
        std_dev = np.std(data)
        print(f"{label}: Mean = {mean:.3f}, Standard Deviation = {std_dev:.3f}")

# Function to toggle visibility of a line in the plot
# Toggle visibility function
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
    # Create the normal distribution lines for each dataset
    for data, label in zip(jaccard_data_list, labels):
        mean = np.mean(data)
        std_dev = np.std(data)

        # Check if standard deviation is zero, skip if true
        if std_dev == 0:
            print(f"Skipping dataset '{label}' due to zero standard deviation.")
            continue

        x = np.linspace(min(data), max(data), 100)
        y = norm.pdf(x, mean, std_dev)

        # Only plot if data is valid
        line, = ax.plot(x, y, label=label)
        lines.append(line)

    ax.set_title("Normal Distribution of Jaccard Similarity")
    ax.set_xlabel("Jaccard Similarity")
    ax.set_ylabel("Density")

    # Place the legend outside the plot on the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add CheckButtons widget outside the plot on the left
    check_ax = plt.axes([0.01, 0.4, 0.2, 0.5])  # Position for the checkboxes on the left of the plot
    check = CheckButtons(check_ax, labels, [True] * len(labels))  # Create checkboxes, all lines visible by default

    # Register callback function to toggle visibility
    check.on_clicked(lambda label: toggle_visibility(label, lines, labels))

    plt.show()

# Function to extract only the time span from a filename (e.g., '2021-2025')
def extract_time_span(filename):
    # Remove 'jaccard_data_' and '.pkl' to extract just the time span
    return os.path.basename(filename).replace('jaccard_data_', '').replace('topic_model_', '').replace('.pkl', '')

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
    
    # Calculate statistics
    calculate_statistics(jaccard_data_list, labels)

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
    table.set_fontsize(12)
    table.scale(1, 2)

    plt.title("Topic Intersection Ratios Table", pad=20)
    plt.show()

## Main execution point

comparison_folder = 'Comparison Results'  # Folder where all the Jaccard data files are stored

# For comparing all results in the 'Comparison Results' folder
compare_results()

# For comparing all specific files
specific_files = ['path/file_1.pkl', 'path/file_2.pkl']    # Update with actual file path
#compare_results(files=specific_files)
