# import torch
# from models.two_view_pipeline import TwoViewPipeline
# from utils.image import ImagePreprocessor
# from utils.tensor import map_tensor
# from kornia import image_to_tensor
# import cv2
# import matplotlib.pyplot as plt
# from visualization.viz2d import plot_images, plot_matches

import torch
from gluefactory.models.two_view_pipeline import TwoViewPipeline
from gluefactory.utils.image import ImagePreprocessor
from gluefactory.utils.tensor import map_tensor
from kornia import image_to_tensor
import cv2
import matplotlib.pyplot as plt
from gluefactory.visualization.viz2d import plot_images, plot_matches

# Path to the best model checkpoint
model_checkpoint_path = r"C:\Users\thoma\OneDrive\2023 Masters\Project\ProjectCode\external\glue-factory\outputs\training\my_experiment\checkpoint_best.tar"
# Paths to the test images
test_image_path_0 = r"C:\Users\thoma\OneDrive\2023 Masters\Project\ProjectCode\external\glue-factory\tests\image_right\000002_right.png"
test_image_path_1 = r"C:\Users\thoma\OneDrive\2023 Masters\Project\ProjectCode\external\glue-factory\tests\image_right\000003_right.png"

# Load the best model checkpoint
checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))

# Initialize the model and load the checkpoint weights
model_conf = checkpoint['conf']['model']
model = TwoViewPipeline(model_conf)
model.load_state_dict(checkpoint['model'])
model.eval()


# Function to prepare input data
def create_input_data(cv_img0, cv_img1, device):
    img0 = image_to_tensor(cv2.cvtColor(cv_img0, cv2.COLOR_BGR2RGB)).float() / 255
    img1 = image_to_tensor(cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)).float() / 255
    ip = ImagePreprocessor({})
    data = {"view0": ip(img0), "view1": ip(img1)}
    data = map_tensor(
        data,
        lambda t: (
            t[None].to(device)
            if isinstance(t, torch.Tensor)
            else torch.from_numpy(t)[None].to(device)
        ),
    )
    return data


# Load test images
cv_img0 = cv2.imread(test_image_path_0)
cv_img1 = cv2.imread(test_image_path_1)

# Create input data
device = "cuda" if torch.cuda.is_available() else "cpu"
data = create_input_data(cv_img0, cv_img1, device)

# Make predictions
with torch.no_grad():
    pred = model(data)

# Convert predictions to numpy for visualization
pred = map_tensor(pred, lambda t: t.cpu().numpy() if isinstance(t, torch.Tensor) else t)


# Print keypoints, confidence levels, and matches for first image
# print("Keypoints, confidence levels, and matches for the first image:")
# for i in range(len(pred["keypoints0"])):
#     kp = pred["keypoints0"][i]
#     match = pred["matches0"][i]
#     score = pred["matching_scores0"][i]
#     # match_status = 'No Match' if match == -1 else match
#     print(f"Keypoint {i} (Image 1): Coordinates: {kp}, Confidence: {score}, Match: {match}")
#
# # Print keypoints, confidence levels, and matches for the second image
# print("\nKeypoints, confidence levels, and matches for the second image:")
# for i in range(len(pred["keypoints1"])):
#     kp = pred["keypoints1"][i]
#     # match = pred["matches0"][match] # if match != -1 else 'No Match'
#     match = pred["matches0"][i]
#     score = pred["matching_scores0"][i] if match != -1 else 'No Confidence Score'
#     print(f"Keypoint {i} (Image 2): Coordinates: {kp}, Confidence: {score}, Match: {match}")

# Print the number of keypoints for debugging
# print(f"Number of keypoints in image 1: {len(kp0)}")
# print(f"Number of keypoints in image 2: {len(kp1)}")

# Plot keypoints on both images

import numpy as np
def plot_keypoints(image, keypoints, confidence, color='r',  default_color='r', high_conf_color='lime'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = np.squeeze(np.array(keypoints), axis=0)
    confidence = np.squeeze(np.array(confidence), axis=0)
    # keypoints = np.array(keypoints)  # Ensure keypoints are in a NumPy array
    # print("keypoints in plot_keypoints", keypoints.shape, keypoints, sep="\n")
    # plt.scatter(keypoints[:, 0], keypoints[:, 1], c=color, s=10)  # Plot all keypoints

    for kp, conf in zip(keypoints, confidence):
        color = high_conf_color if conf > 0 else default_color
        plt.scatter(kp[0], kp[1], c=color, s=30 if conf > 0 else 10)

    # for kp, conf in zip(keypoints, confidence):
    #     print(conf)
    #     if conf > 0:
    #         # color = high_conf_color if conf > 0 else default_color
    #         plt.scatter(kp[0], kp[1], c=high_conf_color, s=30)
    #     else:
    #         plt.scatter(kp[0], kp[1], c=default_color, s=10)

    # plt.show()
    # for kp in keypoints:
    #     plt.plot(kp[0], kp[1], 'o', color=color)
    # plt.show()

# Extract keypoints and matches
print("prediction keys", pred.keys())
kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
matches0 = pred["matches0"]
valid0 = matches0 != -1
kpm0, kpm1 = kp0[valid0], kp1[matches0[valid0]]
confidence = pred["matching_scores0"]
confidence1 = pred["matching_scores1"]

print("kp0:", kp0, "\n\nkp1:", kp1, "\n\nmatches0:", matches0, "\n\nvalid0:", valid0, "\n\nkpm0:", kpm0, "\n\nkpm1:", kpm1, "\n\nconfidence:", confidence)
#
# confidence = np.squeeze(np.array(confidence), axis=0)
# confidence1 = np.squeeze(np.array(confidence1), axis=0)
# kp0tmp = np.squeeze(np.array(kp0), axis=0)
# kp1tmp = np.squeeze(np.array(kp1), axis=0)
#
# for i in range(len(confidence)):
#     if confidence[i] > 0:
#         print(f"confidence for {i} is {confidence[i]} and kp0 point is {kp0tmp[i]}")
#     if confidence1[i] > 0:
#         print(f"confidence for {i} is {confidence1[i]} and kp1 point is {kp1tmp[i]}")
#         # print(confidence[i])
#         # print(i)
        # print(kp0[i], kp1[i])

# Plot images
# plot_images([cv_img0, cv_img1])

# Plot images with keypoints
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plot_keypoints(cv_img0, kp0, confidence, default_color='r', high_conf_color='lime')
# plt.imshow(cv2.cvtColor(cv_img0, cv2.COLOR_BGR2RGB))
# plt.scatter(np.squeeze(kp0)[:, 0], np.squeeze(kp0)[:, 1], c='r', s=10)
plt.title('Image 1 Keypoints')

plt.subplot(1, 2, 2)
plot_keypoints(cv_img1, kp1, confidence1, default_color='g', high_conf_color='lime')
# plt.imshow(cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB))
# plt.scatter(np.squeeze(kp1)[:, 0], np.squeeze(kp1)[:, 1], c='g', s=10)
plt.title('Image 2 Keypoints')

# plt.show()

# Plot matches if there are any
if len(kpm0) > 0 and len(kpm1) > 0:
    print("Found some matches!")
    plot_matches(kpm0, kpm1, a=0.0)

plt.show()

#
# # Plot keypoints on the first image
# print("\nPlotting keypoints on the first image...")
# plot_keypoints(cv_img0, kp0, color='r')
#
# # Plot keypoints on the second image
# print("\nPlotting keypoints on the second image...")
# plot_keypoints(cv_img1, kp1, color='g')
#
# # Plot matches if there are any
# if len(kpm0) > 0 and len(kpm1) > 0:
#     plot_matches(kpm0, kpm1, a=0.0)
#     plt.show()

# # Plot images and matches
# plot_images([cv_img0, cv_img1])
# plot_matches(kpm0, kpm1, a=0.0)
# plt.show()
