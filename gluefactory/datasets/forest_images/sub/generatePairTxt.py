import os
import random


def generate_sequences(image_dir, sequence_length=5):
    images = [img for img in os.listdir(image_dir) if img.endswith('.png')]
    images.sort()  # Ensure the images are sorted in sequential order

    sequences = [images[i:i + sequence_length] for i in range(0, len(images), sequence_length)]
    return sequences


def split_sequences(sequences, train_ratio=0.8):
    random.shuffle(sequences)
    split_idx = int(len(sequences) * train_ratio)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    return train_sequences, val_sequences


def generate_pairs(sequences, output_file):
    with open(output_file, 'w') as f:
        for seq in sequences:
            for i in range(len(seq) - 1):
                img1, img2 = seq[i], seq[i + 1]
                f.write(f"forest_images/{img1} forest_images/{img2}\n")


image_dir = 'C:/Users/thoma/OneDrive/2023 Masters/Project/ProjectCode/external/glue-factory/gluefactory/datasets/forest_images/sub'
sequence_length = 5

# Generate sequences
sequences = generate_sequences(image_dir, sequence_length)

# Split sequences into training and validation sets
train_sequences, val_sequences = split_sequences(sequences)

# Generate pairs for training and validation
generate_pairs(train_sequences, '../train_pairs.txt')
generate_pairs(val_sequences, '../valid_pairs.txt')

print(f"Generated {len(train_sequences)} sequences for training.")
print(f"Generated {len(val_sequences)} sequences for validation.")
