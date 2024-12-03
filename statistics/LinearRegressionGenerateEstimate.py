import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# Function to get user input for sample sizes
def get_sample_sizes():
    choice = input(
        "Would you like to use preset sample sizes (200, 400, 600, 800, 1000) or define your own set of five "
        "increasing sample sizes? (Enter 'preset' or 'custom'): ").strip().lower()

    if choice == 'preset':
        return [200, 400, 600, 800, 1000]
    elif choice == 'custom':
        print("Please enter five increasing sample sizes greater than 2 and within the range 1 to 1000:")
        custom_sample_sizes = []  # Renamed variable to avoid shadowing
        for index in range(5):  # Renamed 'i' to 'index' to avoid shadowing
            while True:
                try:
                    custom_size = int(input(f"Enter sample size {index + 1}: "))  # Renamed 'size' to 'custom_size'
                    if 2 < custom_size <= 1000 and (index == 0 or custom_size > custom_sample_sizes[-1]):
                        custom_sample_sizes.append(custom_size)
                        break
                    else:
                        print(
                            "Invalid input. Ensure that the sample size is greater than the previous one, greater "
                            "than 2, and within the range 1 to 1000.")
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
        return custom_sample_sizes  # Return the custom sample sizes
    else:
        print("Invalid input. Defaulting to preset sample sizes.")
        return [200, 400, 600, 800, 1000]


# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 1000  # Total number of observations
x = np.random.uniform(0, 10, n)  # X is randomly generated over [0, 10]
beta0 = 5
beta1 = -2
noise = np.random.normal(0, 1, n)
y = beta0 + beta1 * x + noise

# Get sample sizes from the user
sample_sizes_list = get_sample_sizes()  # Renamed variable to avoid shadowing

# Find global min and max of y-values for consistent y-axis limits across all plots
global_min_y = float('inf')
global_max_y = float('-inf')

for sample_size in sample_sizes_list:
    y_sample = y[:sample_size]
    min_y = y_sample.min()
    max_y = y_sample.max()
    global_min_y = min(global_min_y, min_y)
    global_max_y = max(global_max_y, max_y)

# Round the global min and max to the nearest integers for cleaner y-ticks
global_min_y = np.floor(global_min_y)
global_max_y = np.ceil(global_max_y)

# Define y-ticks at round numbers, with at most 6 ticks
y_ticks = np.linspace(global_min_y, global_max_y, num=min(6, int(global_max_y - global_min_y + 1)))

# Plotting setup
fig, axes = plt.subplots(1, len(sample_sizes_list), figsize=(20, 5), sharey=True)

for plot_index, sample_size in enumerate(sample_sizes_list):  # Renamed 'i' to 'plot_index' and 'size' to 'sample_size'
    # Subset the data
    x_sample = x[:sample_size]
    y_sample = y[:sample_size]

    # Sort the data by x_sample to ensure smooth plotting
    sorted_indices = np.argsort(x_sample)
    x_sample_sorted = x_sample[sorted_indices]
    y_sample_sorted = y_sample[sorted_indices]

    # Fit the OLS model
    X = sm.add_constant(x_sample_sorted)
    model = sm.OLS(y_sample_sorted, X).fit()

    # Generate predictions and confidence intervals
    pred = model.get_prediction(X)
    pred_summary = pred.summary_frame(alpha=0.05)

    # Plot actual vs. fitted
    axes[plot_index].scatter(x_sample_sorted, y_sample_sorted, color='blue', alpha=0.5, label='Actual')
    axes[plot_index].plot(x_sample_sorted, pred_summary['mean'], color='red', label='Fitted', zorder=2)
    axes[plot_index].fill_between(x_sample_sorted, pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'],
                                  color='red',
                                  alpha=0.3, zorder=1)

    # Add text with model info
    textstr = '\n'.join((
        f'Sample size: {sample_size}',
        f'Intercept: {model.params[0]:.2f} (SE: {model.bse[0]:.2f})',
        f'Slope: {model.params[1]:.2f} (SE: {model.bse[1]:.2f})',
        f'R-squared: {model.rsquared:.2f}'
    ))

    axes[plot_index].text(0.05, 0.95, textstr, transform=axes[plot_index].transAxes, fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Labels and ticks with rotation
    axes[plot_index].set_xlabel('x')
    axes[plot_index].set_ylabel('y', rotation=0, labelpad=15)  # Keep the label upright
    axes[plot_index].tick_params(axis='both', which='both', direction='out')

    # Set consistent y-ticks and y-limits
    axes[plot_index].set_yticks(y_ticks)
    axes[plot_index].set_ylim([global_min_y, global_max_y])

    # Rotate y-tick labels to match the y-axis title
    axes[plot_index].set_yticklabels([f'{int(tick)}' for tick in y_ticks], rotation=0, ha='right')

plt.tight_layout()
plt.show()
