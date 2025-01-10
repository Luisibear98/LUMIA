import argparse
import gc
import json
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, initializers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from lumia_utils import (
    load_models,
    get_activations_and_attention,
    build_classifier
)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mia with hidden states",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", type=str, help="dataset name")
    parser.add_argument("-m", "--model", type=str, help="model size")
    parser.add_argument("-s", "--split", type=str, help="anagram split")
    parser.add_argument("-g", "--gpu", type=str, help="gpu")
    parser.add_argument("-r", "--round", type=str, help="round")
    args = parser.parse_args()
    config = vars(args)

    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda:0")
    print(f"Training model {args.model}")
    model_name = args.model
    model_size = 0
    start_layer = 0
    token = -1
    max_token_division = 2048
    epochs = 1000
    batch_size = 256
    dataset_name = args.dataset
    if "wikimia" in dataset_name:
        split = 32
    else:
        split = args.split
    model_name_file = model_name.replace("/", "")

    file_path = f"./results/{model_name}/{model_name_file}_{dataset_name}_ngram_{split}_{args.round}.csv"

    if not os.path.exists(file_path):
        model_name_dir = model_name.replace("/", "")
        directory = f"./processed_data/{model_name_dir}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        model, tokenizer = load_models(model_name, model_size)

        if "wikimia" in dataset_name:
            wikimia = load_dataset(
                "swj0419/WikiMIA", split=f"WikiMIA_length{split}"
            )
            inputs = wikimia["input"]
            laberu = wikimia["label"]
            members = []
            notMembers = []
            dataset_name = "wikipediamia"
            for i in range(len(laberu)):
                if laberu[i] == 1:
                    members.append(inputs[i])
                else:
                    notMembers.append(inputs[i])
            list_members = members
            list_nonmembers = notMembers

        elif "gutenberg" in dataset_name:
            import os
            from datasets import load_dataset
            from random import sample

            pg19 = load_dataset("imperial-cpg/project-gutenberg-extended")
            list_files = os.listdir("./data/pg19/train")

            nonmember = pg19["train"]["text"]
            limitnonmember = []
            for nomem in nonmember:
                if len(nomem) > 1000:
                    limitnonmember.append(nomem[0:1000])
                else:
                    limitnonmember.append(nomem)
            member = sample(list_files, len(nonmember))

            def read_member(txt):
                base_path = "./data/pg19/train/"
                f = open(f"{base_path}{txt}", "r")
                text = f.read()
                f.close()
                if len(text) > 1000:
                    return text[0:1000]
                else:
                    return text

            member_texts = []
            for members in member:
                info = read_member(members)

                member_texts.append(info)
            list_members = member_texts[0:1000]
            list_nonmembers = limitnonmember[0:1000]

        elif "arxiv-1-month" in dataset_name:
            import os
            from datetime import datetime

            threshold_date = datetime(2023, 3, 1)
            import json

            # Path to your JSONL file

            list_files = os.listdir(
                "./data/pajama/arxiv/"
            )
            members = []
            nonmembers = []
            for file in list_files:
                # Reading the JSONL file
                with open(
                    f"./data/pajama/arxiv/{file}", "r"
                ) as file:
                    for line in file:
                        try:
                            data = json.loads(
                                line
                            )  # Parse each line as a JSON object
                            if (
                                datetime.fromisoformat(data["meta"]["timestamp"])
                                < threshold_date
                            ):
                                members.append(data["text"])
                            else:
                                nonmembers.append(data["text"])
                        except:
                            continue

            import random

            list_members = random.sample(members, len(nonmembers))
            list_nonmembers = nonmembers
        
        elif 'arxiv_mia_cs' in dataset_name:
            data_jsons = []
            data_members = []
            data_nomembers = []
            with open(
                    f"./data/arxiv_mia/arxiv_mia.jsonl", "r"
                ) as file:
                for line in file:
                    data = json.loads(line)
                    if 'cs' in data['field']:
                        data_jsons.append(data)
                        if data['label']:
                            data_members.append(data['text'])
                        else:
                            data_nomembers.append(data['text'])
        elif 'arxiv_mia_math' in dataset_name:
            data_jsons = []
            data_members = []
            data_nomembers = []
            with open(
                    f"./data/arxiv_mia/arxiv_mia.jsonl", "r"
                ) as file:
                for line in file:
                    data = json.loads(line)
                    if 'math' in data['field']:
                        data_jsons.append(data)
                        if data['label']:
                            data_members.append(data['text'])
                        else:
                            data_nomembers.append(data['text'])

            list_members = data_members
            list_nonmembers = data_nomembers
            
        if "pythia" in model_name:
            end_layer = len(model.gpt_neox.layers)
            n_layers = len(model.gpt_neox.layers)

        else:
            end_layer = len(model.transformer.h)
            n_layers = len(model.transformer.h)
        activations_member = []
        activations_nonmember = []

        for member in tqdm(list_members, desc="Processing list_members"):
            split_into = math.floor(len(member) / max_token_division)
            partial_activations = []
            partial_attentions = []
            split_into = 1

            if split_into == 0:
                split_into = 1
            for i in range(split_into):
                init_pos = max_token_division * i
                finish_pos = max_token_division * (i + 1)
                hidden = get_activations_and_attention(
                    model,
                    tokenizer,
                    member,
                    start_layer,
                    end_layer,
                    token,
                    max_token_division,
                )

                partial_activations.append(hidden)
            stacked_tensors = torch.stack(partial_activations, dim=0)
            mean = torch.mean(stacked_tensors, dim=0)
            activations_member.append(mean)

        activations_member = np.array(activations_member)

        emb_dim = activations_member[0].shape[2]

        for nonmember in tqdm(list_nonmembers, desc="Processing len_nonmembers"):
            split_into = math.floor(len(nonmember) / max_token_division)
            partial_activations = []
            partial_attentions = []
            split_into = 1
            if split_into == 0:
                split_into = 1
            for i in range(split_into):
                init_pos = max_token_division * i
                finish_pos = max_token_division * (i + 1)
                try:
                    hidden = get_activations_and_attention(
                        model,
                        tokenizer,
                        nonmember,
                        start_layer,
                        end_layer,
                        token,
                        max_token_division,
                    )

                    partial_activations.append(hidden)

                except:
                    continue

            stacked_tensors = torch.stack(partial_activations, dim=0)
            mean = torch.mean(stacked_tensors, dim=0)
            activations_nonmember.append(mean)
        activations_nonmember = np.array(activations_nonmember)

        if len(activations_member) > len(activations_nonmember):
            activations_member = activations_member[0 : len(activations_nonmember)]
        else:
            activations_nonmember = activations_nonmember[
                0 : len(activations_member)
            ]

       
        activations_member_copy = np.array(activations_member)
        activations_nonmember_copy = np.array(activations_nonmember)

        activations_combined = np.concatenate(
            (activations_member_copy, activations_nonmember_copy)
        )

        labels_member = [1] * len(activations_member)
        labels_nonmember = [0] * len(activations_nonmember)
        labels_combined = labels_member + labels_nonmember
        labels_combined = np.array(labels_combined)

        X_train, X_test, y_train, y_test = train_test_split(
            activations_combined,
            labels_combined,
            test_size=0.2,
            stratify=labels_combined,
        
        )


        max_acc = 0
        max_auc = 0
        acc_arr = []
        acc_auc = []
        layer_arr = []
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        with tf.device('/gpu:1'):
            for layer in range(n_layers):
                print(f"Extracting for layer {layer}")
                X_train_layer = []
                X_test_layer = []

                for sample in X_train:
                    X_train_layer.append(np.array(sample[layer][0]))
                for sample in X_test:
                    X_test_layer.append(np.array(sample[layer][0]))

                num_labels = 2
                X_train_layer = np.array(X_train_layer)
                X_test_layer = np.array(X_test_layer)

                y_train = np.array(y_train)
                y_test = np.array(y_test)




                classifier = build_classifier(emb_dim)

                early_stopping = EarlyStopping(
                    monitor="val_accuracy",  
                    patience=10,
                    restore_best_weights=True,  
                )

                history = classifier.fit(
                    x=X_train_layer,
                    y=y_train,
                    validation_data=(X_test_layer, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0,  
                )

               
                val_accuracies = history.history["val_accuracy"]


                best_val_accuracy = max(val_accuracies)

                max_acc = max(best_val_accuracy, max_acc)
                acc_arr.append(best_val_accuracy)

                layer_arr.append(layer)
                y_pred_probs = classifier.predict(X_test_layer)

                if (
                    len(y_test.shape) == 1 or y_test.shape[1] == 1
                ):  

                    if y_pred_probs.shape[1] == 2:
                        y_pred_probs = y_pred_probs[:, 1]

                    auc_roc = roc_auc_score(y_test, y_pred_probs)
                    print(f"AUC-ROC Score: {auc_roc:.4f}")
                    max_auc = max(max_auc, auc_roc)
                    print(f"MAX AUC-ROC Score: {max_auc:.4f}")
                    acc_auc.append(auc_roc)
                else:
                    
                    print("Multi-class AUC-ROC calculation is not shown here.")

       


        df = pd.DataFrame()
        df["layer"] = layer_arr
        df["acc"] = acc_arr
        df["auc"] = acc_auc
        model_name = model_name.replace("/", "")
        df.to_csv(
            f"./results/{model_name}/{model_name}_{dataset_name}_{split}_{args.round}.csv",
            index=False,
        )

