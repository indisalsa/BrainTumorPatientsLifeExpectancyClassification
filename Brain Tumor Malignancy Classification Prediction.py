import os
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('C:/Users/Lab129/indisalsa/ResNet50V2_Tesis_SMOTE_10.model')


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((112, 112))  # Ensure resize uses a tuple
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


class_labels = ['HGG', 'LGG']
class_weights = {'HGG': 1, 'LGG': 4}  # Assign specific weights to classes


def predict_image_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    return predictions[0]  # Return the raw prediction scores


def process_folders(root_folder):
    results = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg')):
                image_path = os.path.join(root, file)
                prediction_scores = predict_image_class(image_path)
                results.append([os.path.basename(root), file, prediction_scores])
    return results


def weighted_majority_vote(group):
    weighted_scores = {label: 0 for label in class_labels}
    for _, row in group.iterrows():
        for i, label in enumerate(class_labels):
            weighted_scores[label] += row['Scores'][i] * class_weights[label]
    return max(weighted_scores, key=weighted_scores.get)


def save_predictions_to_csv(results, output_file):
    df = pd.DataFrame(results, columns=['BraTS_2020_subject_ID', 'File_Name', 'Scores'])

    # Weighted majority vote
    df['Scores'] = df['Scores'].apply(lambda x: np.array(x))
    majority_vote = df.groupby('BraTS_2020_subject_ID').apply(weighted_majority_vote).reset_index()
    majority_vote.columns = ['BraTS_2020_subject_ID', 'Grade']

    # Writing both to a single CSV with a delimiter
    with open(output_file, 'w', newline='') as file:
        df.to_csv(file, index=False)
        file.write('\n---\n')  # Delimiter row
        majority_vote.to_csv(file, index=False)


# Usage
root_folder = 'C:/Users/Lab129/Downloads/Indi/Dataset SP 128'
results = process_folders(root_folder)
save_predictions_to_csv(results, 'combined_predictions_112_smote_weighted41.csv')
