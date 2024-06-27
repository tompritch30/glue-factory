import pandas as pd

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

# Specify the path to the pickle file
pickle_file_path = '/vol/bitbucket/tp4618/SuperGlueThesis/external/glue-factory/outputs/training/sp+lg_homography/None_Train_20240627_104708.pkl'

# Load the DataFrame from the pickle file
df = pd.read_pickle(pickle_file_path)

# Print the entire DataFrame to see its contents
print("DataFrame Contents:")
print(df)

# Print the columns of the DataFrame to verify the structure
print("\nDataFrame Columns:")
print(df.columns)

# Optionally, print the first few rows to get a quick overview without flooding the console
print("\nFirst few rows of the DataFrame:")
print(df.head())
