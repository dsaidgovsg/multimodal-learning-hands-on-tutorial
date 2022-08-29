import json
import os
import random
from time import perf_counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, get_scheduler

from vl_model import create_model
from ast import literal_eval

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VLDataset(Dataset):
    def __init__(
        self,
        df,
        label_to_id,
        train=False,
        text_field="text",
        label_field="label",
        image_path_field=None,
        image_model_type=None,
        geoloc_start_ind=None,
        geoloc_end_ind=None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.image_path_field = image_path_field
        self.image_model_type = image_model_type
        self.geoloc_start = geoloc_start_ind
        self.geoloc_end = geoloc_end_ind

        # text only dataset
        if image_model_type is not None:

            # ResNet-50 and ALBEF use different image sizes

            if image_model_type.lower() == "resnet":
                # ResNet-50 settings
                self.img_size = 224

            elif image_model_type.lower() == "albef":
                # ALBEF settings
                self.img_size = 256

            self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (
                0.26862954,
                0.26130258,
                0.27577711,
            )

            self.train_transform_func = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

            self.eval_transform_func = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
        label = self.label_to_id[self.df.at[index, self.label_field]]

        if self.geoloc_start is not None and self.geoloc_end is not None:
            # geolocs = self.df.iloc[index, self.geoloc_start: self.geoloc_end].to_numpy()
            geolocs = torch.tensor(
                self.df.iloc[index, self.geoloc_start : self.geoloc_end],
                dtype=torch.float32,
            )

        else:
            geolocs = None

        # return images only if image model is specified
        if self.image_model_type is not None:
            img_path = literal_eval(self.df.at[index, self.image_path_field])[0]

            image = Image.open(img_path)
            if self.train:
                img = self.train_transform_func(image)
            else:
                img = self.eval_transform_func(image)

            if geolocs is not None:
                return text, label, img, geolocs

            return text, label, img

        else:
            if geolocs is not None:
                return text, label, geolocs

            return text, label

    def __len__(self):
        return self.df.shape[0]


class VLClassifier:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        image_model_type=None,
        label_map=None,
        pretrained="bert-base-uncased",
    ):
        self.model = model
        self.tokenizer = (
            AutoTokenizer.from_pretrained(pretrained)
            if tokenizer is None
            else tokenizer
        )
        self.image_model_type = image_model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_id = label_map
        self.id_to_label = (
            {v: k for k, v in self.label_to_id.items()}
            if self.label_to_id is not None
            else None
        )
        if self.model is not None:
            self.model.to(self.device)

    def train(self, df_train, training_args):
        self.training_args = training_args
        pretrained = training_args.get("pretrained")
        batch_size = training_args.get("batch_size")
        num_train_epochs = training_args.get("num_train_epochs")
        learning_rate = training_args.get("learning_rate")
        weight_decay = training_args.get("weight_decay")
        warmup_steps = training_args.get("warmup_steps")
        max_seq_length = training_args.get("max_seq_length")
        text_field = training_args.get("text_field")
        label_field = training_args.get("label_field")
        image_path_field = training_args.get("image_path_field")
        geoloc_start = training_args.get("geoloc_start_index")
        geoloc_end = training_args.get("geoloc_end_index")
        lr_scheduler = training_args.get("lr_scheduler", "cosine")
        if geoloc_start is not None and geoloc_end is not None:
            num_geoloc_features = geoloc_end - geoloc_start
        else:
            num_geoloc_features = 0

        # albef_directory = training_args.get('albef_pretrained_folder', None)

        self.label_to_id = {
            lab: i for i, lab in enumerate(df_train[label_field].unique())
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

        self.model = create_model(
            self.image_model_type,
            self.num_labels,
            text_pretrained=pretrained,
            num_geoloc_features=num_geoloc_features,
        )
        self.model.to(self.device)

        train_dataset = VLDataset(
            df=df_train,
            label_to_id=self.label_to_id,
            train=True,
            text_field=text_field,
            label_field=label_field,
            image_path_field=image_path_field,
            image_model_type=self.image_model_type,
            geoloc_start_ind=geoloc_start,
            geoloc_end_ind=geoloc_end,
        )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=train_sampler
        )

        t_total = len(train_dataloader) * num_train_epochs

        optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total,
        )

        tr_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        self.model.train()
        start = perf_counter()
        for epoch_num in trange(num_train_epochs, desc="Epochs"):
            epoch_total_loss = 0

            for step, batch in tqdm(
                enumerate(train_dataloader), total=len(train_dataloader), desc="Batch"
            ):
                if self.image_model_type is None:
                    b_imgs = None

                    if geoloc_start is not None and geoloc_end is not None:
                        b_text, b_labels, b_geolocs = batch
                    else:
                        b_text, b_labels = batch
                        b_geolocs = None

                else:
                    if geoloc_start is not None and geoloc_end is not None:
                        b_text, b_labels, b_imgs, b_geolocs = batch

                    else:
                        b_text, b_labels, b_imgs = batch
                        b_geolocs = None

                b_inputs = self.tokenizer(
                    list(b_text),
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                    padding=True,
                )

                b_labels = b_labels.to(self.device)
                b_inputs = b_inputs.to(self.device)

                if b_imgs is not None:
                    b_imgs = b_imgs.to(self.device)

                if b_geolocs is not None:
                    b_geolocs = b_geolocs.to(self.device)

                self.model.zero_grad()

                if b_imgs is None:
                    b_logits = self.model(text=b_inputs, geoloc=b_geolocs)
                else:
                    b_logits = self.model(text=b_inputs, image=b_imgs, geoloc=b_geolocs)

                loss = criterion(b_logits, b_labels)

                epoch_total_loss += loss.item()

                # Perform a backward pass to calculate the gradients
                loss.backward()

                optimizer.step()
                scheduler.step()

            tr_loss += epoch_total_loss
            avg_loss = epoch_total_loss / len(train_dataloader)

            print("epoch =", epoch_num)
            print("    epoch_loss =", epoch_total_loss)
            print("    avg_epoch_loss =", avg_loss)
            print("    learning rate =", optimizer.param_groups[0]["lr"])

        end = perf_counter()
        self.training_time = end - start
        print("Training completed in ", self.training_time, "seconds")

    def predict(self, df_test, eval_args):
        batch_size = eval_args.get("batch_size")
        max_seq_length = eval_args.get("max_seq_length")
        text_field = eval_args.get("text_field")
        image_path_field = eval_args.get("image_path_field")
        label_field = eval_args.get("label_field", None)
        geoloc_start = eval_args.get("geoloc_start_index")
        geoloc_end = eval_args.get("geoloc_end_index")

        prediction_results = []

        test_dataset = VLDataset(
            df=df_test,
            label_to_id=self.label_to_id,
            train=False,
            text_field=text_field,
            label_field=label_field,
            image_path_field=image_path_field,
            image_model_type=self.image_model_type,
            geoloc_start_ind=geoloc_start,
            geoloc_end_ind=geoloc_end,
        )
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, sampler=test_sampler
        )

        predict_start = perf_counter()
        for batch in tqdm(test_dataloader):
            self.model.eval()

            if self.image_model_type is None:
                b_imgs = None
                if geoloc_start is not None and geoloc_end is not None:
                    b_text, b_labels, b_geolocs = batch
                else:
                    b_text, b_labels = batch
                    b_geolocs = None

            else:
                if geoloc_start is not None and geoloc_end is not None:
                    b_text, b_labels, b_imgs, b_geolocs = batch

                else:
                    b_text, b_labels, b_imgs = batch
                    b_geolocs = None

            b_inputs = self.tokenizer(
                list(b_text),
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
                padding=True,
            )

            b_inputs = b_inputs.to(self.device)
            b_labels = b_labels.to(self.device)
            if b_imgs is not None:
                b_imgs = b_imgs.to(self.device)

            if b_geolocs is not None:
                b_geolocs = b_geolocs.to(self.device)

            with torch.no_grad():
                if b_imgs is None:
                    b_logits = self.model(text=b_inputs, geoloc=b_geolocs)
                else:
                    b_logits = self.model(text=b_inputs, image=b_imgs, geoloc=b_geolocs)

                b_logits = b_logits.detach().cpu()

            prediction_results += torch.argmax(b_logits, dim=-1).tolist()
        predict_end = perf_counter()

        self.prediction_time = predict_end - predict_start
        print("Prediction completed in ", self.prediction_time, "seconds")
        prediction_labels = [self.id_to_label[p] for p in prediction_results]

        return prediction_labels

    def save(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        model_sd_filepath = os.path.join(save_directory, "state_dict.pt")
        torch.save(self.model.state_dict(), model_sd_filepath)

        label_map_filepath = os.path.join(save_directory, "label_map.json")
        with open(label_map_filepath, "w") as f:
            json.dump(self.label_to_id, f)

        parameters = self.training_args.copy()
        parameters["image_model_type"] = self.image_model_type
        parameters["num_labels"] = len(self.label_to_id)

        parameters_filepath = os.path.join(save_directory, "parameters.json")
        with open(parameters_filepath, "w") as f:
            json.dump(parameters, f)


def from_pretrained(load_directory):
    label_map_filepath = os.path.join(load_directory, "label_map.json")
    with open(label_map_filepath, "r") as f:
        label_map = json.load(f)

    parameters_filepath = os.path.join(load_directory, "parameters.json")
    with open(parameters_filepath, "r") as f:
        parameters = json.load(f)

    image_model_type = parameters["image_model_type"]
    num_labels = parameters["num_labels"]

    geoloc_start_index = parameters.get("geoloc_start_index")
    geoloc_end_index = parameters.get("geoloc_end_index")

    if geoloc_start_index is not None and geoloc_end_index is not None:
        num_geoloc_features = geoloc_end_index - geoloc_start_index
    else:
        num_geoloc_features = 0

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model_sd_filepath = os.path.join(load_directory, "state_dict.pt")
    model_sd = torch.load(model_sd_filepath, map_location="cpu")

    model = create_model(
        image_model_type=image_model_type,
        num_labels=num_labels,
        num_geoloc_features=num_geoloc_features,
    )
    model.to("cpu")  # load all models in cpu first
    model.load_state_dict(model_sd, strict=True)

    return VLClassifier(
        model=model,
        tokenizer=tokenizer,
        image_model_type=image_model_type,
        label_map=label_map,
    )


def classifier_train_test(df_train, df_test, classifier_type, output_folder, args):
    classifier_to_image_model_map = {
        "bert": None,
        "bert_resnet": "resnet",
        "albef": "albef",
    }

    image_model_type = classifier_to_image_model_map[classifier_type]
    classifier = VLClassifier(image_model_type=image_model_type)
    classifier.train(df_train, args)
    predictions = classifier.predict(df_test, args)
    class_report = classification_report(
        df_test[args.get("label_field")], predictions, output_dict=True
    )
    class_report["total_training_time"] = classifier.training_time
    class_report["total_prediction_time"] = classifier.prediction_time

    with open(output_folder + classifier_type + "_class_report.json", "w") as f:
        json.dump(class_report, f)

    df_out = df_test.copy()
    df_out["prediction"] = predictions
    df_out.to_csv(output_folder + classifier_type + "_predictions.csv", index=False)

    model_save_dir = os.path.join(output_folder, classifier_type)
    os.makedirs(model_save_dir)
    classifier.save(model_save_dir)


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to yaml config"
    )
    parsed_args = parser.parse_args()

    yaml_config_path = parsed_args.config
    with open(yaml_config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    results_folder = yaml_config.get("output_folder")
    os.makedirs(results_folder, exist_ok=True)

    train_path = yaml_config.get("train_path")
    test_path = yaml_config.get("test_path")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    seed_val = 0

    training_args = {
        "pretrained": yaml_config.get("pretrained"),
        "batch_size": yaml_config.get("batch_size"),
        "num_train_epochs": yaml_config.get("num_train_epochs"),
        "learning_rate": yaml_config.get("learning_rate"),
        "weight_decay": yaml_config.get("weight_decay"),
        "warmup_steps": yaml_config.get("warmup_steps"),
        "max_seq_length": yaml_config.get("max_seq_length"),
        "text_field": yaml_config.get("text_field"),
        "label_field": yaml_config.get("label_field"),
        "image_path_field": yaml_config.get("image_path_field"),
        "geoloc_start_index": yaml_config.get("geoloc_start_index"),
        "geoloc_end_index": yaml_config.get("geoloc_end_index"),
        "lr_scheduler": yaml_config.get("lr_scheduler"),
        # 'albef_pretrained_folder': albef_folder
    }

    set_seed(seed_val)

    if training_args.get("image_path_field") is None:
        classifier_train_test(
            df_train,
            df_test,
            classifier_type="bert",
            output_folder=results_folder,
            args=training_args,
        )

    # set_seed(seed_val)
    # classifier_train_test(
    #     df_train,
    #     df_test,
    #     classifier_type="bert_resnet",
    #     output_folder=results_folder,
    #     args=args,
    # )
    # set_seed(seed_val)

    else:

        classifier_train_test(
            df_train,
            df_test,
            classifier_type="albef",
            output_folder=results_folder,
            args=training_args,
        )


if __name__ == "__main__":
    main()
