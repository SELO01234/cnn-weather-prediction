import os
import random
import PIL.ImageOps
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps

# Root directory of images
image_root = "./"

# Destination directory for the prepared dataset
dest = "./prepared_dataset/"
os.makedirs(dest, exist_ok=True)

# Desired size of the images
size = 100

# Batch size for processing images
batch_size = 50

# Class labels
classes = ["Sunny", "Rainy", "Snowy", "Foggy"]

def convert_binary_to_class(label):
    """Converts a binary class matrix to a class vector (integer).

    Args:
        label (numpy.ndarray): A binary class matrix.

    Returns:
        list: A list of class indices.
    """
    new_lbl = []
    for i in range(len(label)):
        new_lbl.append(np.argmax(label[i]))
    return new_lbl

def get_accuracy(v_label, y):
    """Calculates the accuracy by comparing labels. If it's not 100 then it's a problem.

    Args:
        v_label (numpy.ndarray): The predicted labels.
        y (numpy.ndarray): The true labels.

    Returns:
        float: The accuracy as a fraction.
    """
    c = np.sum(np.fromiter((np.array_equal(y[i], v_label[i]) for i in range(len(y))), dtype=bool))
    return c / len(y)

def sep_data(v_data, v_label):
    """Separates data into arrays that store [val_data, val_label] for each class.

    Args:
        v_data (numpy.ndarray): The data to be separated.
        v_label (numpy.ndarray): The corresponding labels.

    Returns:
        tuple: Two lists, one for separated data and one for separated labels.
    """
    vd = [[[], []] for _ in range(4)]
    for i in range(len(v_data)):
        cls = int(v_label[i])
        vd[cls][0].append(v_data[i])
        vd[cls][1].append(cls)
    return [np.array(vd[i][0]) for i in range(4)], [np.array(vd[i][1]) for i in range(4)]


def better_crop(image_path, size, resize_only=False):
    """Crops the image to the desired size, with an option to only resize.

    Args:
        image_path (str): The path to the image file.
        size (int): The desired size for cropping or resizing.
        resize_only (bool): If True, only resizes the image without cropping.

    Returns:
        numpy.ndarray: The cropped or resized image as an array.
    """
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    if image.size[0] >= image.size[1]:
        # Landscape image
        if resize_only:
            base_height = size
            wpercent = (base_height / float(image.size[1]))
            wsize = int((float(image.size[0]) * float(wpercent)))
            image = image.resize((wsize, base_height), Image.Resampling.LANCZOS)
            return np.array(image)
        crops = [
            image.crop((0, 0, size, size)),
            image.crop((image.size[0] - size, 0, image.size[0], size)),
            image.crop((0, image.size[1] - size, size, image.size[1])),
            image.crop((image.size[0] - size, image.size[1] - size, image.size[0], image.size[1]))
        ]
        random_crop = random.choice(crops)
        return np.array(random_crop.resize((size, size), Image.Resampling.LANCZOS))
    else:
        # Portrait image
        if resize_only:
            base_width = size
            wpercent = (base_width / float(image.size[0]))
            hsize = int((float(image.size[1]) * float(wpercent)))
            image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
            return np.array(image)
        crops = [
            image.crop((0, 0, size, size)),
            image.crop((0, image.size[1] - size, size, image.size[1])),
            image.crop((image.size[0] - size, 0, image.size[0], size)),
            image.crop((image.size[0] - size, image.size[1] - size, image.size[0], image.size[1]))
        ]
        random_crop = random.choice(crops)
        return np.array(random_crop.resize((size, size), Image.Resampling.LANCZOS))

def prepare_dataset(image_root, dest, size):
    """Prepares the dataset by cropping and resizing images.

    Args:
        image_root (str): The root directory of the images.
        dest (str): The destination directory for the prepared dataset.
        size (int): The desired size for the images.
    """
    for root, _, files in os.walk(image_root):
        for filename in files:
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, filename)
                cropped_image = better_crop(image_path, size, resize_only=False)
                # Save the cropped image to the destination directory
                dest_path = os.path.join(dest, filename)
                Image.fromarray(cropped_image).save(dest_path)

def shuffle_data(data, label):
    """Shuffles the data and labels together.

    Args:
        data (numpy.ndarray): The data to be shuffled.
        label (numpy.ndarray): The corresponding labels.

    Returns:
        tuple: The shuffled data and labels.
    """
    temp = list(zip(data, label))
    random.shuffle(temp)
    shuffled_data, shuffled_labels = zip(*temp)

    # Ensure uniform dimensions
    for i, img in enumerate(shuffled_data):
        if img.shape != shuffled_data[0].shape:
            print(f"Image at index {i} has a different shape: {img.shape}")

    return np.array(shuffled_data), np.array(shuffled_labels)

def image_to_matrix(image_root, dest, size):
    """Converts images to matrices and saves them as .npy files.

    Args:
        image_root (str): The root directory of the images.
        dest (str): The destination directory for the numpy files.
        size (int): The desired size for the images.
    """
    train_data = []
    train_label = []
    classes_dir = ['sunny_e', 'rainy_e', 'snowy_e', 'foggy_e']

    for cls in classes_dir:
        class_path = os.path.join(image_root, cls)
        print(f"Checking class directory: {class_path}")
        if os.path.isdir(class_path):
            for imageName in os.listdir(class_path):
                image_path = os.path.join(class_path, imageName)
                if os.path.isfile(image_path) and image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    try:
                        img = image_utils.load_img(image_path, target_size=(size, size))
                        img = ImageOps.invert(img.convert('RGB'))  # Ensure the image is in RGB mode
                        img = image_utils.img_to_array(img)
                        if img.shape == (size, size, 3):
                            train_data.append(img)
                            train_label.append(classes_dir.index(cls))
                        else:
                            print(f"Skipping {image_path}, incorrect shape: {img.shape}")
                    except Exception as e:
                        print(f"Error processing file {image_path}: {e}")
                else:
                    print(f"'{image_path}' is not a valid image file.")
        else:
            print(f"'{class_path}' is not a directory.")

    if train_data:
        train_data, train_label = shuffle_data(train_data, train_label)
        np.save(os.path.join(dest, "train_data.npy"), np.array(train_data))
        np.save(os.path.join(dest, "train_label.npy"), np.array(train_label))

def images_to_matrix(image_root, dest, size, batch_size=8000):
    """Converts images to matrices and saves them in batches as .npy files.

    Args:
        image_root (str): The root directory of the images.
        dest (str): The destination directory for the numpy files.
        size (int): The desired size for the images.
        batch_size (int): The number of images to process in each batch.
    """
    all_data = []
    all_labels = []
    classes_dir = ['sunny_e', 'rainy_e', 'snowy_e', 'foggy_e']

    for cls in classes_dir:
        class_dir = os.path.join(image_root, cls)
        print(f"Checking class directory: {class_dir}")
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                if os.path.isfile(image_path) and image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    try:
                        img = image_utils.load_img(image_path, target_size=(size, size))
                        img = img.convert('RGB')  # Ensure the image is in RGB mode
                        img = ImageOps.invert(img)
                        img = image_utils.img_to_array(img)
                        if img.shape == (size, size, 3):
                            all_data.append(img)
                            all_labels.append(classes_dir.index(cls))
                        else:
                            print(f"Skipping {image_path}, incorrect shape: {img.shape}")
                    except Exception as e:
                        print(f"Error processing file {image_path}: {e}")
                else:
                    print(f"'{image_path}' is not a valid image file.")
        else:
            print(f"'{class_dir}' is not a directory.")

    if all_data and all_labels:
        all_data, all_labels = shuffle_data(all_data, all_labels)
        np.save(os.path.join(dest, "train_data.npy"), np.array(all_data))
        np.save(os.path.join(dest, "train_label.npy"), np.array(all_labels))
        print(
            f"Saved train_data.npy and train_label.npy with shapes {np.array(all_data).shape} and {np.array(all_labels).shape}")
    else:
        print("No data to save.")

def concatenate_datasets(data_dir="models", output_filename="train_data_concat.npy",
                         label_filename="train_label_concat.npy", max_samples_per_class=None):
    """Concatenates multiple datasets into a single dataset.

    Args:
        data_dir (str): The directory containing the datasets.
        output_filename (str): The name of the output file for the concatenated data.
        label_filename (str): The name of the output file for the concatenated labels.
        max_samples_per_class (int, optional): The maximum number of samples per class.
    """
    all_data = []
    all_labels = []

    for filename in os.listdir(data_dir):
        if (filename.startswith("train_data") and filename.endswith(".npy") and
                filename != output_filename):
            data_path = os.path.join(data_dir, filename)
            label_path = os.path.join(data_dir, filename.replace("data", "label"))

            try:
                data = np.load(data_path)
                labels = np.load(label_path)
                print(f"Loaded {data_path} with shape {data.shape}")
                print(f"Loaded {label_path} with shape {labels.shape}")
            except Exception as e:
                print(f"Error loading file {data_path} or {label_path}: {e}")
                continue

            if max_samples_per_class:
                class_counts = np.bincount(labels)
                indices_to_keep = []
                for class_idx in range(len(class_counts)):
                    class_indices = np.where(labels == class_idx)[0]
                    if len(class_indices) > max_samples_per_class:
                        class_indices = np.random.choice(class_indices, max_samples_per_class, replace=False)
                    indices_to_keep.extend(class_indices)

                data = data[indices_to_keep]
                labels = labels[indices_to_keep]

            all_data.append(data)
            all_labels.append(labels)

    if all_data and all_labels:
        train_data = np.concatenate(all_data, axis=0)
        train_labels = np.concatenate(all_labels, axis=0)
        print(f"Concatenated data shape: {train_data.shape}")
        print(f"Concatenated labels shape: {train_labels.shape}")

        train_data, train_labels = shuffle_data(train_data, train_labels)

        np.save(os.path.join(data_dir, output_filename), train_data)
        np.save(os.path.join(data_dir, label_filename), train_labels)
    else:
        print("No data to concatenate.")


def process_all_images(image_root, dest, size, batch_size):
    #prepare_dataset(image_root, dest, size)
    #images_to_matrix(image_root, dest, size, batch_size)
    train_data = np.load(os.path.join(dest, "train_data.npy"))
    train_label = np.load(os.path.join(dest, "train_label.npy"))
    shuffled_data, shuffled_labels = shuffle_data(train_data, train_label)
    separated_data, separated_labels = sep_data(shuffled_data, shuffled_labels)
   
    combined_data = {
        'separated_data': separated_data,
        'separated_labels': separated_labels
    }
    np.save(os.path.join(dest, "combined_data.npy"), combined_data)
    accuracy = get_accuracy(shuffled_labels, convert_binary_to_class(shuffled_labels))
    print(f"Accuracy of validation after shuffling: {accuracy * 100:.2f}%")

process_all_images(image_root, dest, size, batch_size)
