import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def compare_metrics(df1, df2, metrics, model1, model2):
    results = {}
    for metric in metrics:
        data1 = df1[metric]
        data2 = df2[metric]

        # Calculate means and standard deviations
        mean1, mean2 = data1.mean(), data2.mean()
        std1, std2 = data1.std(), data2.std()

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

        # Store results
        results[metric] = {
            'mean1': mean1, 'std1': std1,
            'mean2': mean2, 'std2': std2,
            't_stat': t_stat, 'p_value': p_value
        }

        # Print results
        print(f"\nMetric: {metric}")
        print(f"Mean {metric} for {model1}: {mean1:.4f}, Std Dev: {std1:.4f}")
        print(f"Mean {metric} for {model2}: {mean2:.4f}, Std Dev: {std2:.4f}")
        print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    return results

def plot_comparison(df1, df2, metrics, image_path):
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(10, 5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]  # Make sure axes is iterable

    for ax, metric in zip(axes, metrics):
        ax.hist(df1[metric], alpha=0.5, label='Model 1 (RGB)', bins=20, edgecolor='black')
        ax.hist(df2[metric], alpha=0.5, label='Model 2 (Grayscale)', bins=20, edgecolor='black')
        ax.set_title(f'Comparison of {metric}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    path = f"{image_path}/testingcomparison_plot.jpg"
    plt.savefig(path)
    print(f"saved image to {path}")


def compare_final_values(df1, df2, metrics, model1_name, model2_name, image_path):
    # Extract the last row from both dataframes
    final_df1 = df1.iloc[-1]
    final_df2 = df2.iloc[-1]
    
    # Create a DataFrame to hold the results
    result_data = {
        model1_name: [],
        model2_name: [],
        'Mean Difference': [],
        'p-value': []
    }
    
    # Compare metrics
    for metric in metrics:
        val1 = final_df1[metric]
        val2 = final_df2[metric]
        
        # Simple difference and mock "p-value" since statistical test not possible with single sample per group
        mean_diff = val1 - val2
        pseudo_p_value = "NA"  # Not applicable in this context

        result_data[model1_name].append(round(val1, 4))
        result_data[model2_name].append(round(val2, 4))
        result_data['Mean Difference'].append(round(mean_diff, 4))
        result_data['p-value'].append(pseudo_p_value)
    
    result_df = pd.DataFrame(result_data, index=metrics)
    result_df.to_latex(f"{image_path}summary_table")
    print(f"saved to {image_path}summary_table.tex")
    
   # Plotting the results table as an image
    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(metrics)))  # Adjust size dynamically
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=result_df.values, colLabels=result_df.columns, rowLabels=result_df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(result_df.columns))))  # Adjust column widths
    
    final_path =f'{image_path}final_comparison_table.jpg' 
    plt.savefig(final_path)
    plt.close()

    print(f"Comparison table saved to {final_path}")
    return result_df

# List of metrics to compare
metrics = ['match_recall', 'match_precision', 'accuracy', 'average_precision',
           'loss/total', 'loss/nll_neg', 'loss/num_matchable', 'loss/num_unmatchable', 'loss/row_norm']
# Load the data
df_rgb = pd.read_pickle("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/rgbDenseHomog.pkl")
df_gray = pd.read_pickle("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/grayscaleDenseHomog.pkl")

model1, model2 = "RGB model", "Grayscale Model"
image_path = f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/'

# Comparing the models
# compare_results = compare_metrics(df_rgb, df_gray, metrics, model1, model2)
# print(compare_results)

compare_final_values(df_rgb, df_gray, metrics, model1, model2, image_path)
# plot_comparison(df_rgb, df_gray, metrics, image_path)
