import numpy as np
import matplotlib.pyplot as plt

def load_and_compare_matrices():
    # Paths to the saved matrices
    """
    Calculated overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_calculated_overlap.npy
    Full overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_full_overlap.npy
    Ground truth overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_ground_truth_overlap.npy
    """
    raw_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_calculated_overlap.npy"
    calculated_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_full_overlap.npy"
    ground_truth_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_ground_truth_overlap.npy"

    # Load matrices
    calculated_overlap_matrix = np.load(raw_path)
    calculated_matrix = np.load(calculated_path)
    ground_truth_matrix = np.load(ground_truth_path)

     # Replace zeros with -1 in both matrices to handle non-calculated overlaps
    calculated_matrix[calculated_matrix == 0] = -1
    ground_truth_matrix[ground_truth_matrix == 0] = -1

    # Determine the maximum shape for a unified matrix dimension
    max_rows = max(calculated_matrix.shape[0], ground_truth_matrix.shape[0])
    max_cols = max(calculated_matrix.shape[1], ground_truth_matrix.shape[1])

    # Adjust matrices to the same shape
    adjusted_calculated = np.full((max_rows, max_cols), -1.0)  # Use -1.0 to initialize
    adjusted_ground_truth = np.full((max_rows, max_cols), -1.0)  # Use -1.0 to initialize

    # Copy the data into the adjusted matrices
    adjusted_calculated[:calculated_matrix.shape[0], :calculated_matrix.shape[1]] = calculated_matrix
    adjusted_ground_truth[:ground_truth_matrix.shape[0], :ground_truth_matrix.shape[1]] = ground_truth_matrix

    # Print shapes and comparison result
    print(f"Adjusted Calculated Matrix Shape: {adjusted_calculated.shape}")
    print(f"Adjusted Ground Truth Matrix Shape: {adjusted_ground_truth.shape}")

    comparison = np.isclose(adjusted_calculated, adjusted_ground_truth, atol=0.01, equal_nan=True)
    print("Comparison Result (True means close enough):")
    print(comparison)

    # Find and print indices where comparison is False
    false_indices = np.where(comparison == False)
    # Iterate over the indices and print the values from both matrices
    for idx in zip(false_indices[0], false_indices[1]):
        calc_value = adjusted_calculated[idx]
        truth_value = adjusted_ground_truth[idx]
        print(f"Index: {idx}, Calculated Value: {calc_value}, Ground Truth Value: {truth_value}")


    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(adjusted_calculated, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Adjusted Calculated Overlap Matrix')

    plt.subplot(1, 2, 2)
    plt.imshow(adjusted_ground_truth, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Adjusted Ground Truth Overlap Matrix')

    plt.show()

    # Print shapes and comparison result

    # full_overlap_matrix = np.full((calculated_overlap_matrix.shape[0], calculated_overlap_matrix.shape[1]), -1.0)
    #     # Populate the full matrix using the mask to place calculated overlaps correctly
    # indices = np.where(mask)[0]  # Get indices where mask is True
    # for i, idx_i in enumerate(indices):
    #     for j, idx_j in enumerate(indices):
    #         if i <= j:  # Fill upper triangle and diagonal
    #             full_overlap_matrix[idx_i, idx_j] = calculated_overlap_matrix[i, j]
    #             if idx_i != idx_j:  # Fill lower triangle if not on the diagonal
    #                 full_overlap_matrix[idx_j, idx_i] = calculated_overlap_matrix[i, j]

    # print(f"calculated_overlap_matrix Matrix Shape: {calculated_overlap_matrix.shape}")
    # print(f"Calculated Matrix Shape: {calculated_matrix.shape}")
    # print(f"Ground Truth Matrix Shape: {ground_truth_matrix.shape}")
    # # Default val should ahve been -1?
   
    # print()
    # # print(calculated_overlap_matrix)
    # print()
    # print(calculated_matrix)
    # print()
    # print(ground_truth_matrix)
    # print()
    # comparison = np.isclose(calculated_matrix, ground_truth_matrix, atol=0.01)
    # print("Comparison Result (True means close enough):")
    # print()
    
    # # Visualization
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(calculated_matrix, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Calculated Overlap Matrix')

    # plt.subplot(1, 2, 2)
    # plt.imshow(ground_truth_matrix, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Ground Truth Overlap Matrix')

    # plt.show()

if __name__ == "__main__":
    load_and_compare_matrices()
