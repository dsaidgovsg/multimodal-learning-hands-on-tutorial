import json
import os

import pandas as pd
from sklearn.metrics import classification_report

from multimodal_training import from_pretrained, set_seed


def main():
    df_test = pd.read_csv(
        "/home/shared/MSO/image_train_test/full_dataset_chatbot/agency/test_with_images.csv"
    )
    seed_val = 0

    # load pretrained ALBEF model and predict on test set
    set_seed(seed_val)
    albef_folder = "/home/shared/MSO/image_train_test/full_dataset_chatbot/agency/results/text_geolocs_images_albef/albef/"
    with open(albef_folder + "parameters.json", "r") as f:
        albef_args = json.load(f)

    albef_classifier = from_pretrained(albef_folder)
    albef_predictions = albef_classifier.predict(df_test, albef_args)
    albef_class_report = classification_report(
        df_test[albef_args.get("label_field")], albef_predictions, output_dict=True
    )
    print(albef_class_report)
    print("ALBEF Accuracy:", albef_class_report["accuracy"])


if __name__ == "__main__":
    main()
