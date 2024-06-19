import numpy as np
import matplotlib.pyplot as plt

def load_and_compare_matrices():
    # Paths to the saved matrices
    """
    Calculated overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_calculated_overlap.npy
    Full overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_full_overlap.npy
    Ground truth overlap matrix saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_ground_truth_overlap.npy
    """
    # raw_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_calculated_overlap.npy"
    calculated_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_full_overlap.npy"
    ground_truth_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/0204_ground_truth_overlap.npy"

    # Load matrices
    # calculated_overlap_matrix = np.load(raw_path)
    calculated_matrix = np.load(calculated_path)
    ground_truth_matrix = np.load(ground_truth_path)

    # Print shapes and comparison result
    # print(f"calculated_overlap_matrix Matrix Shape: {calculated_overlap_matrix.shape}")
    print(f"Calculated Matrix Shape: {calculated_matrix.shape}")
    print(f"Ground Truth Matrix Shape: {ground_truth_matrix.shape}")
    # Default val should ahve been -1?
   
    print()
    # print(calculated_overlap_matrix)
    print()
    print(calculated_matrix)
    print()
    print(ground_truth_matrix)
    print()
    comparison = np.isclose(calculated_matrix, ground_truth_matrix, atol=0.01)
    print("Comparison Result (True means close enough):")
    print()

    # Find indices where comparison is False and print corresponding values
    false_indices = np.where(comparison == False)
    print("Indices and Values where comparison is False:")
    for idx in zip(false_indices[0], false_indices[1]):
        calc_value = calculated_matrix[idx]
        truth_value = ground_truth_matrix[idx]
        print(f"Index: {idx}, Calculated Value: {calc_value}, Ground Truth Value: {truth_value}")

        
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
