from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import pyperclip

def process(csv_files):
    # Initialize empty lists to accumulate data
    all_sizes = []
    all_times = []

    # Loop through each CSV file
    for csv_file in (csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Extract columns for plotting
        size = df["SIZE"]
        time = df["TIME(microseconds)"]

        # Accumulate data
        all_sizes.extend(size)
        all_times.extend(time)

        # Plot data with a unique color for each file
        plt.plot(size, time, marker='o', linestyle='-', label=f'{csv_file}')

    # Set plot labels and title
    plt.xlabel("SIZE")
    plt.ylabel("TIME (microseconds)")
    plt.title("Combined Plot for All Files")

    # Set precision on the y-axis
    precision = 0  # Set the desired precision
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter(f"{{x:.{precision}f}}"))

    # Enable hover text for the y-values
    cursor = mplcursors.cursor(hover=True)

    def onclick(sel):
        y_value = sel.target[1]
        pyperclip.copy(f"{y_value:.{precision}f}")

    cursor.connect("add", onclick)

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
