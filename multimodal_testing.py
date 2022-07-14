import json
import os

import pandas as pd
from sklearn.metrics import classification_report

from multimodal_training import from_pretrained, set_seed


def main():
    home_folder = "./KDD/"
    trained_model_folders = home_folder + "trained_models/"
    data_folder = home_folder + "webvision_data/"
    image_folder = data_folder + "images/"
    results_folder = home_folder + "results/"
    os.makedirs(results_folder, exist_ok=True)

    df_test = pd.read_csv(data_folder + "test.csv")
    df_test["img_path"] = df_test["img_path"].apply(lambda x: image_folder + x)

    seed_val = 0
    set_seed(seed_val)

    # load pretrained bert model and predict on test set

    bert_folder = trained_model_folders + "BERT/"
    with open(bert_folder + "parameters.json", "r") as f:
        bert_args = json.load(f)

    bert_classifier = from_pretrained(bert_folder)
    bert_predictions = bert_classifier.predict(df_test.copy(), bert_args)
    bert_class_report = classification_report(
        df_test[bert_args.get("label_field")], bert_predictions, output_dict=True
    )
    print(bert_class_report)
    print("BERT Accuracy:", bert_class_report["accuracy"])

    # load pretrained bert-resnet model and predict on test set
    set_seed(seed_val)
    bert_resnet_folder = trained_model_folders + "BERT_ResNet/"
    with open(bert_resnet_folder + "parameters.json", "r") as f:
        bert_resnet_args = json.load(f)

    bert_resnet_classifier = from_pretrained(bert_resnet_folder)
    bert_resnet_predictions = bert_resnet_classifier.predict(
        df_test.copy(), bert_resnet_args
    )
    bert_resnet_class_report = classification_report(
        df_test[bert_resnet_args.get("label_field")],
        bert_resnet_predictions,
        output_dict=True,
    )
    print(bert_resnet_class_report)
    print("BERT-ResNet Accuracy:", bert_resnet_class_report["accuracy"])

    # load pretrained ALBEF model and predict on test set
    set_seed(seed_val)
    albef_folder = trained_model_folders + "ALBEF/"
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
