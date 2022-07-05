import json
import os
import torch
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from vl_model import create_model
import random
import numpy as np
from multimodal_training import VLClassifier, from_pretrained


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def main():
    home_folder = './KDD/'
    trained_model_folders = home_folder + 'trained_models/'
    data_folder = home_folder + 'webvision_data/'
    image_folder = data_folder + 'images/'
    results_folder = home_folder + 'results/'
    os.makedirs(results_folder, exist_ok=True)

    df_test = pd.read_csv(data_folder + 'test.csv')

    seed_val = 0

    # bert_folder = trained_model_folders + 'BERT/'
    # with open(bert_folder + 'parameters.json', 'r') as f:
    #     bert_args = json.load(f)


    bert_resnet_folder = trained_model_folders + 'BERT_ResNet/'
    with open(bert_resnet_folder + 'parameters.json', 'r') as f:
        bert_resnet_args = json.load(f)


    # albef_folder = trained_model_folders + 'ALBEF/'
    # with open(albef_folder + 'parameters.json', 'r') as f:
    #     albef_args = json.load(f)

    
    df_test['img_path'] = df_test['img_path'].apply(lambda x: image_folder + x)
    print(df_test.head())


    
    # bert_classifier = from_pretrained(bert_folder)
    # predictions = bert_classifier.predict(df_test, bert_args)
    # class_report = classification_report(df_test[bert_args.get('label_field')], predictions, output_dict=True)
    # print(class_report)



    bert_resnet_classifier = from_pretrained(bert_resnet_folder)
    predictions = bert_resnet_classifier.predict(df_test, bert_resnet_args)
    class_report = classification_report(df_test[bert_resnet_args.get('label_field')], predictions, output_dict=True)
    print(class_report)



    # albef_classifier = from_pretrained(albef_folder)
    # predictions = albef_classifier.predict(df_test, albef_args)
    # class_report = classification_report(df_test[albef_args.get('label_field')], predictions, output_dict=True)
    # print(class_report)




if __name__ == "__main__":
    main()

