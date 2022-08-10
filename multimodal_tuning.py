import optuna
import argparse
import yaml
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score


from multimodal_training import VLClassifier, set_seed


class Tuner:
    def __init__(self, yaml_config):
        self.yaml_config = yaml_config
        pass

    def objective(self, trial):
        epochs = trial.suggest_int("epochs", 0, 20)
        learning_rate = trial.suggest_loguniform("learning_rate", 2.0e-7, 2.0e-3)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        lr_scheduler = trial.suggest_categorical("lr_scheduler", ["linear", "cosine"])

        train_path = self.yaml_config.get("train_path")
        validation_path = self.yaml_config.get("validation_path")
        # test_path = yaml_config.get("test_path")

        image_path_field = self.yaml_config.get("image_path_field")
        label_field = self.yaml_config.get("label_field")
        pretrained = self.yaml_config.get("pretrained")
        max_seq_length = self.yaml_config.get("max_seq_length")
        weight_decay = self.yaml_config.get("weight_decay")
        warmup_steps = self.yaml_config.get("warmup_steps")
        text_field = self.yaml_config.get("text_field")
        label_field = self.yaml_config.get("label_field")
        image_path_field = self.yaml_config.get("image_path_field")
        geoloc_start_index = self.yaml_config.get("geoloc_start_index")
        geoloc_end_index = self.yaml_config.get("geoloc_end_index")

        df_train = pd.read_csv(train_path)
        df_validation = pd.read_csv(validation_path)
        # df_test = pd.read_csv(test_path)

        seed_val = 0

        fixed_args = {
            "pretrained": pretrained,
            "max_seq_length": max_seq_length,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "text_field": text_field,
            "label_field": label_field,
            "image_path_field": image_path_field,
            "geoloc_start_index": geoloc_start_index,
            "geoloc_end_index": geoloc_end_index,
        }

        tunable_args = {
            "batch_size": batch_size,
            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "lr_scheduler": lr_scheduler,
        }

        training_args = {**fixed_args, **tunable_args}

        set_seed(seed_val)

        image_model_type = None if image_path_field is None else "albef"
        classifier = VLClassifier(image_model_type=image_model_type)
        classifier.train(df_train, training_args)
        predictions = classifier.predict(df_validation, training_args)
        validation_labels = df_validation[label_field], predictions

        # return best accuracy
        return accuracy_score(validation_labels, predictions)

    def run(self, n_trials):
        # study = optuna.create_study(direction="maximize", storage=self.storage_path)
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        return study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to yaml config"
    )
    args = parser.parse_args()

    yaml_config_path = args.config
    with open(yaml_config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    n_trials = yaml_config.get("n_trials")

    results_folder = yaml_config.get("output_folder")
    os.makedirs(results_folder, exist_ok=True)

    tuner = Tuner(yaml_config)
    study = tuner.run(n_trials=n_trials)
    all_trials = study.get_trials()
    sorted_trials = [
        str(ft) for ft in sorted(all_trials, key=lambda x: x.values[0], reverse=True)
    ]
    with open(os.path.join(results_folder, "sorted_trials.json"), "w") as f:
        json.dump(sorted_trials, f)

    best_params = study.best_params
    with open(os.path.join(results_folder, "tuning_best_params.json"), "w") as f:
        json.dump(best_params, f)


if __name__ == "__main__":
    main()
