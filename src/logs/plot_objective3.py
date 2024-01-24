import matplotlib.pyplot as plt

def plot(data_dict, name):
    # Extract X and Y values from the dictionary
    x_values = list(data_dict.keys())
    y_values = list(data_dict.values())
    x_values = sorted(x_values, key=lambda x: int(x))

    # Plotting a bar chart
    _, ax = plt.subplots()
    bars = ax.bar(x_values, y_values, align='center')

    # Adding labels above the bars
    for bar in bars:
        base_height = bars[0].get_height()
        height = bar.get_height()
        formatted_height = '{:.1e}'.format(height)
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{formatted_height}({round(base_height/height, 2)}x)', ha='center', va='bottom')

    # Adding labels and title
    ax.set_xlabel('MPISIZE')
    ax.set_ylabel('microseconds')
    ax.set_title(name)

    # Display the plot
    plt.show()
