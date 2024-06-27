import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

"""
Validation DataFrame saved to: /vol/bitbucket/tp4618/SuperGlueThesis/external/glue-factory/outputs/training/sp+lg_homography/None_Validation_20240627_104708.pkl
['epoch', 'iteration', 'match_recall', 'match_precision', 'accuracy',
       'average_precision', 'loss/total', 'loss/last', 'loss/assignment_nll',
       'loss/nll_pos', 'loss/nll_neg', 'loss/num_matchable',
       'loss/num_unmatchable', 'loss/row_norm']

epoch iteration  match_recall  match_precision  accuracy  average_precision  ...  loss/assignment_nll  loss/nll_pos  loss/nll_neg  loss/num_matchable  loss/num_unmatchable  loss/row_norm
0     0         0      0.001347         0.207941  0.595700           0.000570  ...             4.590368      8.375255      0.805480           203.15047            296.898903       0.469402
1     0       318      0.681990         0.662553  0.811850           0.501607  ...             1.541371      2.352543      0.730198           203.15047            296.898903       0.672488

Train DataFrame saved to: /vol/bitbucket/tp4618/SuperGlueThesis/external/glue-factory/outputs/training/sp+lg_homography/None_Train_20240627_104708.pkl
 epoch  iteration     total      last  assignment_nll   nll_pos   nll_neg  num_matchable  num_unmatchable  confidence  row_norm
0      0          0  6.718346  5.428674        5.428674  8.891179  1.966169        182.125         317.0625    0.546830  0.158397
1      0        100  2.318268  1.520358        1.520358  2.056605  0.984112        242.000         259.6250    0.486482  0.551142
2      0        200  3.181324  2.583654        2.583654  4.516363  0.650945        193.500         307.1250    0.408713  0.678297
3      0        300  2.355447  1.742110        1.742110  2.773631  0.710589        196.750         304.6250    0.373123  0.629811

['epoch', 'iteration', 'total', 'last', 'assignment_nll', 'nll_pos',
       'nll_neg', 'num_matchable', 'num_unmatchable', 'confidence',
       'row_norm']

"""

def plot_metrics(df, metrics, base_image_dir, filename):
    os.makedirs(base_image_dir, exist_ok=True)

    for metric in metrics:
        safe_metric_name = metric.replace('/', '_')
        plt.figure()
        plt.plot(df['iteration'] + df['epoch'] * 1000, df[metric], label=f'{metric} per iteration')
        plt.xlabel('Iteration (Epoch x 1000 + Iteration)')
        plt.ylabel(metric)
        plt.title(f'Plot of {metric}')
        plt.legend()
        plt.grid(True)
        image_path = f'{base_image_dir}/{safe_metric_name}_{filename}_plot.jpg'
        plt.savefig(image_path)
        print(f'Individual plot saved as: {image_path}')
        plt.close()


def create_multiplot(image_paths, titles, output_filename, filename):
    num_images = len(image_paths)
    cols = 2  # Define the number of columns for your grid
    rows = (num_images + cols - 1) // cols  # Calculate the necessary number of rows

    plt.figure(figsize=(cols * 5, rows * 5))

    for i, (path, title) in enumerate(zip(image_paths, titles)):
        img = mpimg.imread(path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Combined plot saved as {output_filename}")

#### Globals #### 
dataframe_path = "/vol/bitbucket/tp4618/SuperGlueThesis/external/glue-factory/outputs/training/sp+lg_homography/None_Validation_20240627_104708.pkl"
# will take this None_Train_20240627_104708
filename = dataframe_path.split("/")[-1].split(".")[0]
# Set basdir 
base_image_dir = f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/{filename}"
df = pd.read_pickle(dataframe_path)

#### Accuracy
metrics = ['match_recall', 'match_precision', 'accuracy', 'average_precision']
plot_metrics(df, metrics, base_image_dir, filename)

# Example usage
titles = metrics  # Assuming titles are the same as the metric names
exp = "Accuracy_Metrics"
image_paths = [f'{base_image_dir}/{metric}_{filename}_plot.jpg' for metric in metrics]
output_filename = f"{base_image_dir}/{exp}_combined_metrics.jpg"
create_multiplot(image_paths, titles, output_filename, filename)
# print(f"Saved multiplot to {output_filename}")

##### Loss

print("Now doing loss metrics")
metrics = ['loss/total', 'loss/assignment_nll', 'loss/num_matchable', 'loss/num_unmatchable']
plot_metrics(df, metrics, base_image_dir, filename)

# Example usage
titles = metrics  
exp = "Loss"
# Separate out the replacement to see if it resolves the syntax error
metrics_modified = [metric.replace('/', '_') for metric in metrics]
image_paths = [f'{base_image_dir}/{metric}_{filename}_plot.jpg' for metric in metrics_modified]
output_filename = f"{base_image_dir}/{exp}_combined_metrics.jpg"
create_multiplot(image_paths, titles, output_filename, filename)
# print(f"Saved multiplot to {output_filename}")

