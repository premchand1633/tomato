{
  "cells": [
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "4Z2Hwax_N__A"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import joblib\n",
        "from fastai.vision.all import *\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "\n",
        "def load_and_resize_images(folder_path, max_images=30):\n",
        "    print(f\"Loading images from '{folder_path}'\")\n",
        "\n",
        "    # Get a list of image paths in the folder\n",
        "    image_paths = get_image_files(folder_path)[:max_images]\n",
        "\n",
        "    # Resize and save each image\n",
        "    for img_path in image_paths:\n",
        "        img = Image.open(img_path)\n",
        "        img = img.resize((192, 192))  # Resize images to the desired dimensions\n",
        "        img.save(img_path)\n",
        "\n",
        "    return image_paths\n",
        "\n",
        "# Load and resize images for ripe and unripe tomatoes from local dataset\n",
        "ripe_image_paths = load_and_resize_images('local_dataset/ripe_tomato')\n",
        "unripe_image_paths = load_and_resize_images('local_dataset/unripe_tomato')\n",
        "\n",
        "# Convert images to numpy arrays\n",
        "ripe_imgs = [np.array(Image.open(img)).reshape(-1) for img in ripe_image_paths]\n",
        "unripe_imgs = [np.array(Image.open(img)).reshape(-1) for img in unripe_image_paths]\n",
        "\n",
        "# Create X and y arrays for logistic regression\n",
        "X = np.vstack([ripe_imgs, unripe_imgs])\n",
        "y = np.array([1] * len(ripe_imgs) + [0] * len(unripe_imgs))  # 1 for ripe, 0 for unripe\n",
        "\n",
        "# Split data into training and validation sets\n",
        "X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Train the logistic regression model\n",
        "model = LogisticRegression()\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Make predictions on the validation set\n",
        "y_pred = model.predict(X_valid)\n",
        "\n",
        "# Evaluate the model\n",
        "accuracy = accuracy_score(y_valid, y_pred)\n",
        "print(f\"Validation Accuracy: {accuracy:.2f}\")\n",
        "\n",
        "# Save the model\n",
        "joblib.dump(model, 'ripe_or_not_logistic_regression_local.pkl')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wBkLh8F8NmU_",
        "outputId": "e8cba10d-8b2b-4cad-868e-fb082eb38c6b"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loading images from 'local_dataset/ripe_tomato'\n",
            "Loading images from 'local_dataset/unripe_tomato'\n",
            "Validation Accuracy: 0.92\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['ripe_or_not_logistic_regression_local.pkl']"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "COEDgZ93N4OF"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.5"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}