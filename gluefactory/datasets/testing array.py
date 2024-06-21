import numpy as np

# Define the individual matrices
pose1 = np.array([[-0.99972585, 0.00444084, 0.02298924, -0.582747],
                  [-0.0025746, 0.95504571, -0.2964474, -0.482966],
                  [-0.02327225, -0.29642532, -0.95477245, -1.49953],
                  [0.0, 0.0, 0.0, 1.0]])

pose2 = np.array([[-0.86847403, 0.00488399, -0.49571062, -3.27473],
                  [0.16254231, 0.94747838, -0.27543551, -0.877939],
                  [0.46832987, -0.31978254, -0.82365421, -2.38428],
                  [0.0, 0.0, 0.0, 1.0]])

pose3 = np.array([[-0.99972948, 0.00437591, 0.02284325, -0.583693],
                  [-0.00257747, 0.95524887, -0.29579206, -0.478772],
                  [-0.02311535, -0.29577093, -0.95497918, -1.51625],
                  [0.0, 0.0, 0.0, 1.0]])

# Create an empty array of objects
# poses = np.array([], dtype=object)

# # Append each pose to the array
# poses = np.append(poses, np.array(pose1, dtype=object))
# poses = np.append(poses, np.array(pose2, dtype=object))
# poses = np.append(poses, np.array(pose3, dtype=object))

# Create an empty array of objects
# poses = np.empty((0,), dtype=object)

# # Append each entire pose matrix to the array
# poses = np.append(poses, [pose1])
# poses = np.append(poses, [pose2])
# poses = np.append(poses, [pose3])

# print(f"Type of poses: {type(poses)}")
# print(f"Shape of poses: {poses.shape}")
# for i, pose in enumerate(poses, start=1):
#     print(f"Pose {i}:")
#     print(pose)

poses = np.array(np.array([[pose1, pose2, pose3]], dtype=object))

print(f"Type of poses: {type(poses)}")
print(f"Shape of poses: {poses.shape}")
print(poses)
for i, pose in enumerate(poses, start=1):
    print(f"Pose {i}:")
    print(pose)