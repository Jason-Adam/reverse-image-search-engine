import pickle
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors


filenames = pickle.load(open("training/data/filenames-caltech101-resnet.pkl", "rb"))
feature_list = pickle.load(open("training/data/features-caltech101-resnet.pkl", "rb"))
num_images = len(filenames)


# Helper function to get the classname
def classname(str):
    return str.split("/")[-2]


# Helper function to get the classname and filename
def classname_filename(str):
    return str.split("/")[-2] + "/" + str.split("/")[-1]


# Helper functions to plot the nearest images given a query image
def plot_images(filenames, distances):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20, 10))
    columns = 4
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + classname_filename(filenames[i]))
        else:
            ax.set_title(
                "Similar Image\n"
                + classname_filename(filenames[i])
                + "\nDistance: "
                + str(float("{0:.2f}".format(distances[i])))
            )
        plt.imshow(image)


def main():
    # Train NearestNeighbors
    neighbors = NearestNeighbors(
        n_neighbors=5, algorithm="brute", metric="euclidean"
    ).fit(feature_list)
    distances, indices = neighbors.kneighbors([feature_list[0]])

    # Visualize
    for _ in range(6):
        random_image_index = random.randint(0, num_images)
        distances, indices = neighbors.kneighbors([feature_list[random_image_index]])

        # Don't take the first closest image as it will be the same image
        similar_image_paths = [filenames[random_image_index]] + [
            filenames[indices[0][i]] for i in range(1, 4)
        ]
        plot_images(similar_image_paths, distances[0])

    plt.show()


if __name__ == "__main__":
    main()
