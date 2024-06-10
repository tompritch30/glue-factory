import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.image as mpimg

def parse_log_data(log_data):
    """
    Parses raw log data into a structured pandas DataFrame.
    
    Args:
        log_data (str): Multiline string containing raw log entries.
    
    Returns:
        pd.DataFrame: DataFrame with structured log data.
    """
    # Regular expression pattern to extract essential elements from each log entry
    # pattern = (
    #     r'\[(?P<date>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) gluefactory INFO\] '  # Capture log date and time
    #     r'\[(E (?P<epoch>\d+) \| it (?P<iteration>\d+))? ?(Validation)?\] '     # Capture epoch, iteration, and check for 'Validation'
    #     r'\{(?P<data>.+)\}'                                                      # Capture the data block in curly braces
    # )
    pattern = (
    r'\[(?P<date>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) gluefactory INFO\] '  # Capture log date and time
    r'\[?(E (?P<epoch>\d+) \| it (?P<iteration>\d+))? ?(Validation)?\]? '     # Optionally match epoch, iteration, and 'Validation' within brackets
    r'\{(?P<data>.+)\}'                                                      # Capture the data block in curly braces
    )
    
    rows = []  # List to store parsed data as dictionaries

    # Iterate through each line of the log data
    for line in log_data.splitlines():
        match = re.search(pattern, line)  # Search for pattern in the line
        if match:
            row_data = match.groupdict()  # Extract matched groups as a dictionary
            print(f"Matched row: {row_data}")  # Debug output

            data_dict = {}
            # Extract key-value pairs from the data part using a regex
            # This captures two possible keys (if present) and the corresponding value formatted as a float in scientific notation
            details = re.findall(r'(\w+)/?(\w+)? (\d+\.\d+E[+-]?\d+)', row_data['data'])
            for key1, key2, value in details:
                # If a second key part exists, concatenate with the first (e.g., 'loss/total'), otherwise use the first key
                if key2:
                    key = f"{key1}/{key2}"
                else:
                    key = key1
                data_dict[key] = float(value.replace('E', 'e'))  # Convert the string value to a float and add to the dictionary

            print(f"Extracted data dict: {data_dict}")  # Debug output

            # Update the original data row with parsed key-value pairs
            row_data.update(data_dict)
            # Remove the raw data string since it's already parsed into individual elements
            row_data.pop('data')
            # Append the updated row data to the list of rows
            rows.append(row_data)
        else:
            print(f"No match for line: {line}")  # Debug for unmatched lines

    # Create a DataFrame from the list of row dictionaries
    df = pd.DataFrame(rows)
    print(f"DataFrame columns before type conversion: {df.columns}")  # Debug output

    # Convert the 'date' column to datetime objects for better manipulation and filtering
    df['date'] = pd.to_datetime(df['date'])
    # Identify columns that are numeric and convert their data types appropriately for analysis
    numeric_cols = df.columns.difference(['date', 'epoch', 'iteration', 'Validation'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    print(f"DataFrame after type conversion: {df.head()}")  # Debug output
    
    return df

def plot_loss(df):
    """ Plots the total loss over iterations grouped by epochs. """
    # Check if 'Validation' column exists and filter out validation rows for loss plotting
    print("\n\n\n", df, df.shape)
    if 'Validation' in df.columns:
        df = df[df['Validation'].isna()]
    
    if 'epoch' in df.columns and 'iteration' in df.columns and 'total' in df.columns:
        for epoch in df['epoch'].unique():
            epoch_data = df[df['epoch'] == epoch]
            plt.plot(epoch_data['iteration'], epoch_data['total'], label=f'Epoch {epoch}')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.title('Total Loss over Iterations by Epoch')
        plt.legend()
        plt.show()
    else:
        print("Required columns are missing in DataFrame")

# Example usage
log_data = """
[06/06/2024 16:00:19 gluefactory INFO] [E 0 | it 0] loss {total 7.121E+00, last 6.084E+00, assignment_nll 6.084E+00, nll_pos 1.023E+01, nll_neg 1.941E+00, num_matchable 1.287E+02, num_unmatchable 3.730E+02, confidence 5.501E-01, row_norm 1.607E-01}
[06/06/2024 16:14:10 gluefactory INFO] [Validation] {match_recall 1.257E-04, match_precision 2.975E-02, accuracy 6.636E-01, average_precision 7.656E-06, loss/total 5.104E+00, loss/last 5.104E+00, loss/assignment_nll 5.104E+00, loss/nll_pos 9.299E+00, loss/nll_neg 9.088E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 4.237E-01}
[06/06/2024 16:14:10 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 16:14:12 gluefactory INFO] New best val: loss/total=5.104108297548588
[06/06/2024 16:17:15 gluefactory INFO] [E 0 | it 100] loss {total 3.204E+00, last 2.531E+00, assignment_nll 2.531E+00, nll_pos 4.057E+00, nll_neg 1.004E+00, num_matchable 1.336E+02, num_unmatchable 3.701E+02, confidence 4.082E-01, row_norm 4.872E-01}
[06/06/2024 16:21:58 gluefactory INFO] [E 0 | it 200] loss {total 2.695E+00, last 1.976E+00, assignment_nll 1.976E+00, nll_pos 3.148E+00, nll_neg 8.033E-01, num_matchable 1.489E+02, num_unmatchable 3.536E+02, confidence 3.669E-01, row_norm 6.024E-01}
[06/06/2024 16:26:56 gluefactory INFO] [E 0 | it 300] loss {total 2.120E+00, last 1.435E+00, assignment_nll 1.435E+00, nll_pos 2.287E+00, nll_neg 5.829E-01, num_matchable 1.716E+02, num_unmatchable 3.317E+02, confidence 3.277E-01, row_norm 7.155E-01}
[06/06/2024 16:31:01 gluefactory INFO] [E 0 | it 400] loss {total 2.185E+00, last 1.497E+00, assignment_nll 1.497E+00, nll_pos 2.446E+00, nll_neg 5.483E-01, num_matchable 1.514E+02, num_unmatchable 3.488E+02, confidence 3.242E-01, row_norm 7.466E-01}
[06/06/2024 16:35:13 gluefactory INFO] [E 0 | it 500] loss {total 2.407E+00, last 1.694E+00, assignment_nll 1.694E+00, nll_pos 2.665E+00, nll_neg 7.244E-01, num_matchable 1.292E+02, num_unmatchable 3.734E+02, confidence 3.019E-01, row_norm 6.786E-01}
[06/06/2024 16:39:21 gluefactory INFO] [Validation] {match_recall 7.110E-01, match_precision 6.314E-01, accuracy 8.447E-01, average_precision 5.045E-01, loss/total 1.352E+00, loss/last 1.352E+00, loss/assignment_nll 1.352E+00, loss/nll_pos 2.070E+00, loss/nll_neg 6.343E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 7.291E-01}
[06/06/2024 16:39:21 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 16:39:23 gluefactory INFO] New best val: loss/total=1.3521605860014543
[06/06/2024 16:43:42 gluefactory INFO] [E 0 | it 600] loss {total 2.163E+00, last 1.476E+00, assignment_nll 1.476E+00, nll_pos 2.297E+00, nll_neg 6.558E-01, num_matchable 1.351E+02, num_unmatchable 3.665E+02, confidence 3.110E-01, row_norm 7.309E-01}
[06/06/2024 16:47:45 gluefactory INFO] [E 0 | it 700] loss {total 2.381E+00, last 1.732E+00, assignment_nll 1.732E+00, nll_pos 2.933E+00, nll_neg 5.314E-01, num_matchable 1.610E+02, num_unmatchable 3.421E+02, confidence 2.775E-01, row_norm 7.602E-01}
[06/06/2024 16:52:09 gluefactory INFO] [E 0 | it 800] loss {total 1.704E+00, last 9.925E-01, assignment_nll 9.925E-01, nll_pos 1.467E+00, nll_neg 5.180E-01, num_matchable 1.502E+02, num_unmatchable 3.508E+02, confidence 2.945E-01, row_norm 7.830E-01}
[06/06/2024 16:56:23 gluefactory INFO] [E 0 | it 900] loss {total 1.922E+00, last 1.219E+00, assignment_nll 1.219E+00, nll_pos 1.893E+00, nll_neg 5.458E-01, num_matchable 1.362E+02, num_unmatchable 3.651E+02, confidence 2.956E-01, row_norm 7.875E-01}
[06/06/2024 17:00:59 gluefactory INFO] [E 0 | it 1000] loss {total 1.967E+00, last 1.370E+00, assignment_nll 1.370E+00, nll_pos 2.179E+00, nll_neg 5.597E-01, num_matchable 1.508E+02, num_unmatchable 3.509E+02, confidence 2.546E-01, row_norm 7.824E-01}
[06/06/2024 17:05:06 gluefactory INFO] [Validation] {match_recall 7.807E-01, match_precision 6.448E-01, accuracy 8.558E-01, average_precision 5.509E-01, loss/total 1.049E+00, loss/last 1.049E+00, loss/assignment_nll 1.049E+00, loss/nll_pos 1.490E+00, loss/nll_neg 6.077E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 7.857E-01}
[06/06/2024 17:05:07 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 17:05:08 gluefactory INFO] New best val: loss/total=1.0487601253437389
[06/06/2024 17:12:53 gluefactory INFO] [Validation] {match_recall 7.921E-01, match_precision 6.450E-01, accuracy 8.564E-01, average_precision 5.567E-01, loss/total 9.955E-01, loss/last 9.955E-01, loss/assignment_nll 9.955E-01, loss/nll_pos 1.436E+00, loss/nll_neg 5.547E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 7.884E-01}
[06/06/2024 17:12:53 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 17:12:54 gluefactory INFO] New best val: loss/total=0.9955168498589151
[06/06/2024 17:12:56 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_0_1083.tar
[06/06/2024 17:12:58 gluefactory INFO] Starting epoch 1
[06/06/2024 17:12:58 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/06/2024 17:13:04 gluefactory INFO] [E 1 | it 0] loss {total 1.859E+00, last 1.140E+00, assignment_nll 1.140E+00, nll_pos 1.703E+00, nll_neg 5.767E-01, num_matchable 1.371E+02, num_unmatchable 3.646E+02, confidence 3.086E-01, row_norm 7.727E-01}
[06/06/2024 17:16:06 gluefactory INFO] [E 1 | it 100] loss {total 2.207E+00, last 1.559E+00, assignment_nll 1.559E+00, nll_pos 2.545E+00, nll_neg 5.732E-01, num_matchable 1.155E+02, num_unmatchable 3.856E+02, confidence 2.521E-01, row_norm 7.627E-01}
[06/06/2024 17:19:08 gluefactory INFO] [E 1 | it 200] loss {total 1.836E+00, last 1.162E+00, assignment_nll 1.162E+00, nll_pos 1.825E+00, nll_neg 4.994E-01, num_matchable 1.448E+02, num_unmatchable 3.579E+02, confidence 2.628E-01, row_norm 7.925E-01}
[06/06/2024 17:22:10 gluefactory INFO] [E 1 | it 300] loss {total 1.471E+00, last 8.191E-01, assignment_nll 8.191E-01, nll_pos 1.255E+00, nll_neg 3.833E-01, num_matchable 1.718E+02, num_unmatchable 3.319E+02, confidence 2.630E-01, row_norm 8.492E-01}
[06/06/2024 17:25:12 gluefactory INFO] [E 1 | it 400] loss {total 1.486E+00, last 8.272E-01, assignment_nll 8.272E-01, nll_pos 1.249E+00, nll_neg 4.053E-01, num_matchable 1.599E+02, num_unmatchable 3.410E+02, confidence 2.781E-01, row_norm 8.504E-01}
[06/06/2024 17:28:15 gluefactory INFO] [E 1 | it 500] loss {total 1.771E+00, last 1.114E+00, assignment_nll 1.114E+00, nll_pos 1.726E+00, nll_neg 5.025E-01, num_matchable 1.275E+02, num_unmatchable 3.759E+02, confidence 2.513E-01, row_norm 7.949E-01}
[06/06/2024 17:37:38 gluefactory INFO] [Validation] {match_recall 8.087E-01, match_precision 6.734E-01, accuracy 8.738E-01, average_precision 5.880E-01, loss/total 8.750E-01, loss/last 8.750E-01, loss/assignment_nll 8.750E-01, loss/nll_pos 1.273E+00, loss/nll_neg 4.771E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.375E-01}
[06/06/2024 17:37:38 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 17:37:39 gluefactory INFO] New best val: loss/total=0.8749505441026347
[06/06/2024 17:40:43 gluefactory INFO] [E 1 | it 600] loss {total 1.642E+00, last 9.804E-01, assignment_nll 9.804E-01, nll_pos 1.491E+00, nll_neg 4.696E-01, num_matchable 1.389E+02, num_unmatchable 3.624E+02, confidence 2.542E-01, row_norm 8.334E-01}
[06/06/2024 17:44:19 gluefactory INFO] [E 1 | it 700] loss {total 1.996E+00, last 1.361E+00, assignment_nll 1.361E+00, nll_pos 2.209E+00, nll_neg 5.137E-01, num_matchable 1.487E+02, num_unmatchable 3.547E+02, confidence 2.292E-01, row_norm 7.898E-01}
[06/06/2024 17:47:40 gluefactory INFO] [E 1 | it 800] loss {total 1.629E+00, last 9.631E-01, assignment_nll 9.631E-01, nll_pos 1.455E+00, nll_neg 4.714E-01, num_matchable 1.461E+02, num_unmatchable 3.545E+02, confidence 2.575E-01, row_norm 8.299E-01}
[06/06/2024 17:51:14 gluefactory INFO] [E 1 | it 900] loss {total 1.656E+00, last 9.523E-01, assignment_nll 9.523E-01, nll_pos 1.368E+00, nll_neg 5.366E-01, num_matchable 1.320E+02, num_unmatchable 3.676E+02, confidence 2.812E-01, row_norm 8.472E-01}
[06/06/2024 17:54:23 gluefactory INFO] [E 1 | it 1000] loss {total 1.485E+00, last 8.597E-01, assignment_nll 8.597E-01, nll_pos 1.304E+00, nll_neg 4.149E-01, num_matchable 1.604E+02, num_unmatchable 3.402E+02, confidence 2.520E-01, row_norm 8.473E-01}
[06/06/2024 17:58:30 gluefactory INFO] [Validation] {match_recall 8.346E-01, match_precision 6.811E-01, accuracy 8.789E-01, average_precision 6.076E-01, loss/total 7.652E-01, loss/last 7.652E-01, loss/assignment_nll 7.652E-01, loss/nll_pos 1.101E+00, loss/nll_neg 4.298E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.506E-01}
[06/06/2024 17:58:30 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 17:58:31 gluefactory INFO] New best val: loss/total=0.7651892635603202
[06/06/2024 18:05:31 gluefactory INFO] [Validation] {match_recall 8.151E-01, match_precision 6.801E-01, accuracy 8.790E-01, average_precision 5.964E-01, loss/total 8.405E-01, loss/last 8.405E-01, loss/assignment_nll 8.405E-01, loss/nll_pos 1.234E+00, loss/nll_neg 4.470E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.280E-01}
[06/06/2024 18:05:33 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_1_2167.tar
[06/06/2024 18:05:34 gluefactory INFO] Starting epoch 2
[06/06/2024 18:05:34 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/06/2024 18:05:42 gluefactory INFO] [E 2 | it 0] loss {total 1.630E+00, last 1.010E+00, assignment_nll 1.010E+00, nll_pos 1.560E+00, nll_neg 4.606E-01, num_matchable 1.325E+02, num_unmatchable 3.698E+02, confidence 2.466E-01, row_norm 8.171E-01}
[06/06/2024 18:08:56 gluefactory INFO] [E 2 | it 100] loss {total 1.834E+00, last 1.072E+00, assignment_nll 1.072E+00, nll_pos 1.617E+00, nll_neg 5.276E-01, num_matchable 1.267E+02, num_unmatchable 3.748E+02, confidence 2.735E-01, row_norm 7.918E-01}
[06/06/2024 18:11:59 gluefactory INFO] [E 2 | it 200] loss {total 1.244E+00, last 5.620E-01, assignment_nll 5.620E-01, nll_pos 6.710E-01, nll_neg 4.531E-01, num_matchable 1.574E+02, num_unmatchable 3.449E+02, confidence 2.634E-01, row_norm 8.732E-01}
[06/06/2024 18:15:02 gluefactory INFO] [E 2 | it 300] loss {total 1.270E+00, last 5.895E-01, assignment_nll 5.895E-01, nll_pos 8.294E-01, nll_neg 3.497E-01, num_matchable 1.738E+02, num_unmatchable 3.297E+02, confidence 2.668E-01, row_norm 8.824E-01}
[06/06/2024 18:18:07 gluefactory INFO] [E 2 | it 400] loss {total 1.531E+00, last 9.407E-01, assignment_nll 9.407E-01, nll_pos 1.552E+00, nll_neg 3.294E-01, num_matchable 1.546E+02, num_unmatchable 3.470E+02, confidence 2.292E-01, row_norm 8.760E-01}
[06/06/2024 18:21:09 gluefactory INFO] [E 2 | it 500] loss {total 1.955E+00, last 1.322E+00, assignment_nll 1.322E+00, nll_pos 2.175E+00, nll_neg 4.692E-01, num_matchable 1.117E+02, num_unmatchable 3.912E+02, confidence 2.298E-01, row_norm 8.070E-01}
[06/06/2024 18:25:16 gluefactory INFO] [Validation] {match_recall 8.519E-01, match_precision 6.855E-01, accuracy 8.808E-01, average_precision 6.200E-01, loss/total 6.981E-01, loss/last 6.981E-01, loss/assignment_nll 6.981E-01, loss/nll_pos 9.746E-01, loss/nll_neg 4.216E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.644E-01}
[06/06/2024 18:25:16 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 18:25:18 gluefactory INFO] New best val: loss/total=0.6980689649940376
[06/06/2024 18:28:22 gluefactory INFO] [E 2 | it 600] loss {total 1.421E+00, last 8.005E-01, assignment_nll 8.005E-01, nll_pos 1.143E+00, nll_neg 4.576E-01, num_matchable 1.415E+02, num_unmatchable 3.612E+02, confidence 2.479E-01, row_norm 8.427E-01}
[06/06/2024 18:31:25 gluefactory INFO] [E 2 | it 700] loss {total 1.522E+00, last 8.499E-01, assignment_nll 8.499E-01, nll_pos 1.272E+00, nll_neg 4.274E-01, num_matchable 1.627E+02, num_unmatchable 3.397E+02, confidence 2.545E-01, row_norm 8.604E-01}
[06/06/2024 18:34:29 gluefactory INFO] [E 2 | it 800] loss {total 1.279E+00, last 7.295E-01, assignment_nll 7.295E-01, nll_pos 1.065E+00, nll_neg 3.937E-01, num_matchable 1.533E+02, num_unmatchable 3.483E+02, confidence 2.410E-01, row_norm 8.924E-01}
[06/06/2024 18:37:31 gluefactory INFO] [E 2 | it 900] loss {total 1.514E+00, last 8.524E-01, assignment_nll 8.524E-01, nll_pos 1.229E+00, nll_neg 4.761E-01, num_matchable 1.498E+02, num_unmatchable 3.486E+02, confidence 2.725E-01, row_norm 8.752E-01}
[06/06/2024 18:40:34 gluefactory INFO] [E 2 | it 1000] loss {total 1.264E+00, last 6.361E-01, assignment_nll 6.361E-01, nll_pos 8.904E-01, nll_neg 3.819E-01, num_matchable 1.547E+02, num_unmatchable 3.462E+02, confidence 2.451E-01, row_norm 8.880E-01}
[06/06/2024 18:44:50 gluefactory INFO] [Validation] {match_recall 8.628E-01, match_precision 6.882E-01, accuracy 8.829E-01, average_precision 6.282E-01, loss/total 6.520E-01, loss/last 6.520E-01, loss/assignment_nll 6.520E-01, loss/nll_pos 8.714E-01, loss/nll_neg 4.327E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.690E-01}
[06/06/2024 18:44:50 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 18:44:52 gluefactory INFO] New best val: loss/total=0.6520104080517647
[06/06/2024 18:52:48 gluefactory INFO] [Validation] {match_recall 8.584E-01, match_precision 6.961E-01, accuracy 8.874E-01, average_precision 6.329E-01, loss/total 6.595E-01, loss/last 6.595E-01, loss/assignment_nll 6.595E-01, loss/nll_pos 9.415E-01, loss/nll_neg 3.775E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.769E-01}
[06/06/2024 18:52:51 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_2_3251.tar
[06/06/2024 18:52:52 gluefactory INFO] Starting epoch 3
[06/06/2024 18:52:52 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/06/2024 18:52:58 gluefactory INFO] [E 3 | it 0] loss {total 1.337E+00, last 6.815E-01, assignment_nll 6.815E-01, nll_pos 1.017E+00, nll_neg 3.459E-01, num_matchable 1.365E+02, num_unmatchable 3.669E+02, confidence 2.278E-01, row_norm 8.905E-01}
[06/06/2024 18:56:01 gluefactory INFO] [E 3 | it 100] loss {total 2.042E+00, last 1.445E+00, assignment_nll 1.445E+00, nll_pos 2.415E+00, nll_neg 4.740E-01, num_matchable 1.166E+02, num_unmatchable 3.851E+02, confidence 2.166E-01, row_norm 8.006E-01}
[06/06/2024 18:59:04 gluefactory INFO] [E 3 | it 200] loss {total 1.181E+00, last 5.127E-01, assignment_nll 5.127E-01, nll_pos 6.460E-01, nll_neg 3.794E-01, num_matchable 1.499E+02, num_unmatchable 3.531E+02, confidence 2.497E-01, row_norm 8.977E-01}
[06/06/2024 19:02:06 gluefactory INFO] [E 3 | it 300] loss {total 1.174E+00, last 5.685E-01, assignment_nll 5.685E-01, nll_pos 8.034E-01, nll_neg 3.337E-01, num_matchable 1.589E+02, num_unmatchable 3.441E+02, confidence 2.329E-01, row_norm 8.975E-01}
[06/06/2024 19:05:09 gluefactory INFO] [E 3 | it 400] loss {total 1.119E+00, last 5.377E-01, assignment_nll 5.377E-01, nll_pos 7.065E-01, nll_neg 3.690E-01, num_matchable 1.531E+02, num_unmatchable 3.487E+02, confidence 2.265E-01, row_norm 8.888E-01}
[06/06/2024 19:08:12 gluefactory INFO] [E 3 | it 500] loss {total 1.704E+00, last 9.747E-01, assignment_nll 9.747E-01, nll_pos 1.490E+00, nll_neg 4.596E-01, num_matchable 1.124E+02, num_unmatchable 3.896E+02, confidence 2.536E-01, row_norm 8.532E-01}
[06/06/2024 19:12:35 gluefactory INFO] [Validation] {match_recall 8.729E-01, match_precision 6.949E-01, accuracy 8.862E-01, average_precision 6.392E-01, loss/total 6.163E-01, loss/last 6.163E-01, loss/assignment_nll 6.163E-01, loss/nll_pos 7.966E-01, loss/nll_neg 4.360E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.897E-01}
[06/06/2024 19:12:35 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 19:12:36 gluefactory INFO] New best val: loss/total=0.6163047115983923
[06/06/2024 19:15:41 gluefactory INFO] [E 3 | it 600] loss {total 1.188E+00, last 6.077E-01, assignment_nll 6.077E-01, nll_pos 8.083E-01, nll_neg 4.072E-01, num_matchable 1.550E+02, num_unmatchable 3.480E+02, confidence 2.273E-01, row_norm 8.935E-01}
[06/06/2024 19:18:43 gluefactory INFO] [E 3 | it 700] loss {total 1.640E+00, last 1.080E+00, assignment_nll 1.080E+00, nll_pos 1.711E+00, nll_neg 4.498E-01, num_matchable 1.546E+02, num_unmatchable 3.469E+02, confidence 2.095E-01, row_norm 8.456E-01}
[06/06/2024 19:21:46 gluefactory INFO] [E 3 | it 800] loss {total 1.279E+00, last 7.180E-01, assignment_nll 7.180E-01, nll_pos 1.112E+00, nll_neg 3.246E-01, num_matchable 1.450E+02, num_unmatchable 3.568E+02, confidence 2.140E-01, row_norm 9.042E-01}
[06/06/2024 19:25:12 gluefactory INFO] [E 3 | it 900] loss {total 1.350E+00, last 6.777E-01, assignment_nll 6.777E-01, nll_pos 9.378E-01, nll_neg 4.177E-01, num_matchable 1.327E+02, num_unmatchable 3.656E+02, confidence 2.487E-01, row_norm 8.796E-01}
[06/06/2024 19:28:15 gluefactory INFO] [E 3 | it 1000] loss {total 1.054E+00, last 4.817E-01, assignment_nll 4.817E-01, nll_pos 5.930E-01, nll_neg 3.704E-01, num_matchable 1.568E+02, num_unmatchable 3.452E+02, confidence 2.367E-01, row_norm 9.023E-01}
[06/06/2024 19:32:23 gluefactory INFO] [Validation] {match_recall 8.786E-01, match_precision 6.952E-01, accuracy 8.868E-01, average_precision 6.422E-01, loss/total 5.913E-01, loss/last 5.913E-01, loss/assignment_nll 5.913E-01, loss/nll_pos 7.981E-01, loss/nll_neg 3.845E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.796E-01}
[06/06/2024 19:32:23 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 19:32:25 gluefactory INFO] New best val: loss/total=0.5913137092071326
[06/06/2024 19:39:04 gluefactory INFO] [Validation] {match_recall 8.750E-01, match_precision 6.914E-01, accuracy 8.850E-01, average_precision 6.373E-01, loss/total 6.082E-01, loss/last 6.082E-01, loss/assignment_nll 6.082E-01, loss/nll_pos 8.263E-01, loss/nll_neg 3.902E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.723E-01}
[06/06/2024 19:39:06 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_3_4335.tar
[06/06/2024 19:39:08 gluefactory INFO] Starting epoch 4
[06/06/2024 19:39:08 gluefactory INFO] lr changed from 0.0001 to 0.0001
[06/06/2024 19:39:14 gluefactory INFO] [E 4 | it 0] loss {total 1.483E+00, last 9.119E-01, assignment_nll 9.119E-01, nll_pos 1.426E+00, nll_neg 3.975E-01, num_matchable 1.353E+02, num_unmatchable 3.660E+02, confidence 2.236E-01, row_norm 8.555E-01}
[06/06/2024 19:42:16 gluefactory INFO] [E 4 | it 100] loss {total 1.380E+00, last 7.033E-01, assignment_nll 7.033E-01, nll_pos 1.010E+00, nll_neg 3.962E-01, num_matchable 1.284E+02, num_unmatchable 3.715E+02, confidence 2.444E-01, row_norm 8.920E-01}
[06/06/2024 19:45:18 gluefactory INFO] [E 4 | it 200] loss {total 1.063E+00, last 4.118E-01, assignment_nll 4.118E-01, nll_pos 4.523E-01, nll_neg 3.713E-01, num_matchable 1.542E+02, num_unmatchable 3.474E+02, confidence 2.481E-01, row_norm 9.195E-01}
[06/06/2024 19:48:21 gluefactory INFO] [E 4 | it 300] loss {total 1.314E+00, last 7.069E-01, assignment_nll 7.069E-01, nll_pos 1.064E+00, nll_neg 3.503E-01, num_matchable 1.578E+02, num_unmatchable 3.461E+02, confidence 2.276E-01, row_norm 8.826E-01}
[06/06/2024 19:51:24 gluefactory INFO] [E 4 | it 400] loss {total 1.165E+00, last 6.314E-01, assignment_nll 6.314E-01, nll_pos 9.544E-01, nll_neg 3.083E-01, num_matchable 1.590E+02, num_unmatchable 3.430E+02, confidence 2.058E-01, row_norm 9.098E-01}
[06/06/2024 19:54:27 gluefactory INFO] [E 4 | it 500] loss {total 1.531E+00, last 8.384E-01, assignment_nll 8.384E-01, nll_pos 1.263E+00, nll_neg 4.140E-01, num_matchable 1.311E+02, num_unmatchable 3.706E+02, confidence 2.458E-01, row_norm 8.679E-01}
[06/06/2024 19:58:43 gluefactory INFO] [Validation] {match_recall 8.728E-01, match_precision 6.982E-01, accuracy 8.884E-01, average_precision 6.416E-01, loss/total 6.247E-01, loss/last 6.247E-01, loss/assignment_nll 6.247E-01, loss/nll_pos 7.862E-01, loss/nll_neg 4.633E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.952E-01}
[06/06/2024 20:01:48 gluefactory INFO] [E 4 | it 600] loss {total 1.216E+00, last 6.731E-01, assignment_nll 6.731E-01, nll_pos 9.510E-01, nll_neg 3.951E-01, num_matchable 1.405E+02, num_unmatchable 3.604E+02, confidence 2.237E-01, row_norm 8.818E-01}
[06/06/2024 20:03:44 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_4_5000.tar
[06/06/2024 20:04:52 gluefactory INFO] [E 4 | it 700] loss {total 1.666E+00, last 1.008E+00, assignment_nll 1.008E+00, nll_pos 1.621E+00, nll_neg 3.958E-01, num_matchable 1.625E+02, num_unmatchable 3.396E+02, confidence 2.219E-01, row_norm 8.619E-01}
[06/06/2024 20:07:54 gluefactory INFO] [E 4 | it 800] loss {total 1.136E+00, last 5.459E-01, assignment_nll 5.459E-01, nll_pos 7.123E-01, nll_neg 3.795E-01, num_matchable 1.449E+02, num_unmatchable 3.565E+02, confidence 2.185E-01, row_norm 8.957E-01}
[06/06/2024 20:10:58 gluefactory INFO] [E 4 | it 900] loss {total 1.346E+00, last 7.083E-01, assignment_nll 7.083E-01, nll_pos 9.823E-01, nll_neg 4.343E-01, num_matchable 1.354E+02, num_unmatchable 3.635E+02, confidence 2.529E-01, row_norm 8.762E-01}
[06/06/2024 20:14:00 gluefactory INFO] [E 4 | it 1000] loss {total 1.190E+00, last 5.502E-01, assignment_nll 5.502E-01, nll_pos 7.313E-01, nll_neg 3.691E-01, num_matchable 1.521E+02, num_unmatchable 3.491E+02, confidence 2.262E-01, row_norm 8.917E-01}
[06/06/2024 20:18:08 gluefactory INFO] [Validation] {match_recall 8.817E-01, match_precision 7.074E-01, accuracy 8.938E-01, average_precision 6.544E-01, loss/total 5.587E-01, loss/last 5.587E-01, loss/assignment_nll 5.587E-01, loss/nll_pos 7.497E-01, loss/nll_neg 3.677E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.905E-01}
[06/06/2024 20:18:08 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_best.tar
[06/06/2024 20:18:09 gluefactory INFO] New best val: loss/total=0.5586933724218816
[06/06/2024 20:24:49 gluefactory INFO] [Validation] {match_recall 8.781E-01, match_precision 7.099E-01, accuracy 8.953E-01, average_precision 6.550E-01, loss/total 5.677E-01, loss/last 5.677E-01, loss/assignment_nll 5.677E-01, loss/nll_pos 7.841E-01, loss/nll_neg 3.514E-01, loss/num_matchable 1.687E+02, loss/num_unmatchable 3.317E+02, loss/row_norm 8.899E-01}
[06/06/2024 20:24:51 gluefactory.utils.experiments INFO] Saving checkpoint checkpoint_4_5419.tar
[06/06/2024 20:24:53 gluefactory.utils.experiments INFO] Deleting checkpoint checkpoint_0_78.tar
[06/06/2024 20:24:53 gluefactory INFO] Finished training on process 0.
"""


def parse_validation_logs(log_data):
    """
    Parses raw log data to extract validation entries and map them to custom 'epoch_iteration' keys.
    
    Args:
        log_data (str): Multiline string containing raw log entries.
        
    Returns:
        pd.DataFrame: DataFrame containing structured log data for validation entries with custom keys.
    """
    pattern = (
        r'\[(?P<date>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) gluefactory INFO\] '  # Capture log date and time
        r'\[Validation\] '  # Identify validation logs
        r'\{(?P<data>.+)\}'  # Capture the data block in curly braces
    )
    
    rows = []
    validation_counter = 0

    # Define the key sequence based on your specified pattern
    epoch_patterns = ["_0", "500", "1000", "final"] * 4 + ["500", "1000", "final", "final_final"]

    # Iterate through each line of the log data
    for line in log_data.splitlines():
        if "[Validation]" in line:
            match = re.search(pattern, line)
            if match:
                row_data = match.groupdict()
                
                # Determine the custom epoch_iteration label
                epoch_iteration = epoch_patterns[validation_counter]
                
                # Parse data block into key-value pairs
                data_dict = {key: float(value.replace('E', 'e')) for key, value in re.findall(r'(\w+/\w+|\w+) (\d+\.\d+E[+-]?\d+)', row_data['data'])}
                
                # Update row data and assign custom key
                row_data.update(data_dict)
                row_data['epoch_iteration'] = epoch_iteration
                row_data.pop('data', None)  # Remove the original data string
                
                rows.append(row_data)
                validation_counter += 1  # Increment the counter after processing a validation log

    # Create DataFrame
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])  # Convert date string to datetime object
    
    return df

# Function to plot data
def plot_validation_loss(df, dataKey='loss/total'):
    """
    Plots the validation loss from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the log data.
    """
    print("DataFrame shape:", df.shape)
    print(df[['epoch_iteration', 'loss/total']])

    # Adjusting epoch_iteration labels
    epochs_per_cycle = 4
    iteration_per_epoch = 4

    for i in range(len(df)):
        epoch_num = i // iteration_per_epoch
        if i % iteration_per_epoch == 3:
            iteration_label = 'final'
        else:
            iteration_label = df.loc[i, 'epoch_iteration']
        if i == len(df) - 1:
            df.loc[i, 'epoch_iteration'] = 'final_final'
        else:
            df.loc[i, 'epoch_iteration'] = f"{epoch_num}_{iteration_label}"

    if dataKey in df.columns:
        print("Plotting loss total:")
        print("Epoch Iterations:", df['epoch_iteration'].unique())
        print("Loss Total Stats:", df[dataKey].describe())
        
        plt.figure(figsize=(15, 5))
        # plt.plot(df['epoch_iteration'], df['loss/total'], marker='o', linestyle='-', color='b')
        plt.plot(df['epoch_iteration'], df[dataKey], marker='o', linestyle='-', color='b')
        plt.title(f'{dataKey} over Custom Epoch Iterations ')
        plt.xlabel('Epoch_Iteration')
        plt.ylabel(dataKey)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # plt.title('Validation Loss over Custom Epoch Iterations')
        # plt.xlabel('Epoch_Iteration')
        # plt.ylabel('Loss Total')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        dataKey = dataKey.replace("/", "_")
        plt.savefig(f'validation_loss_plot_{dataKey}.jpg')  # Saves the plot as a JPEG file
        plt.close()  # Close the plot to free up memory
        print(f"Plot saved as 'validation_loss_plot_{dataKey}.jpg'.")
        # plt.show()
    else:
        print("Loss total data not available in DataFrame.")


def create_multiplot(image_paths, titles, output_filename):
    """
    Creates a multiplot from given image files and saves it.

    Args:
        image_paths (list of str): List of paths to the image files.
        titles (list of str): Titles for each subplot.
        output_filename (str): Path to save the combined image.
    """
    plt.figure(figsize=(20, 10))  # Adjust size as necessary
    num_images = len(image_paths)
    for i in range(num_images):
        img = mpimg.imread(image_paths[i])
        plt.subplot(2, 2, i + 1)  # Adjust grid dimensions as necessary
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Combined plot saved as {output_filename}")

# # Paths to your images
# image_paths = [
#     '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_accuracy.jpg',
#     '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_match_precision.jpg',
#     '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_match_recall.jpg',
#     '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_average_precision.jpg'
# ]

# # Titles for each subplot
# titles = [
#     'Accuracy', 
#     'Match Precision', 
#     'Match Recall', 
#     'Average Precision'
# ]

# # Filename to save the combined plot
# output_filename = '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/performance_metrics.jpg'

# # Create the multiplot
# create_multiplot(image_paths, titles, output_filename)

# Paths to the loss metric images
loss_image_paths = [
    '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_loss_total.jpg',
    '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_loss_last.jpg',
    '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_loss_nll_pos.jpg',
    '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/validation_loss_plot_loss_nll_neg.jpg'
]

# Titles for each of the loss metric plots
loss_titles = [
    'Total Loss',
    'Last Loss',
    'NLL Positive',
    'NLL Negative'
]

# Filename to save the combined loss metrics plot
loss_output_filename = '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/loss_metrics.jpg'

# Call the function to create and save the multiplot
create_multiplot(loss_image_paths, loss_titles, loss_output_filename)

## make plots on just validaiton
# df = parse_validation_logs(log_data)
# for key in df.columns:
#     plot_validation_loss(df, key)

# df = parse_log_data(log_data)
# plot_loss(df)

## combined plots