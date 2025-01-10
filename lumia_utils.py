import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXModel
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import bitsandbytes as bnb  
from transformers import BitsAndBytesConfig
from baukit import Trace
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers, losses


model_name = ""
def load_models(model_name_aux, model_size):
    model_name = model_name_aux
    print("getting model ")
    if '12b' not in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        model = model.to("cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
    else:
        
        # load model in 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )

        model = GPTNeoXForCausalLM.from_pretrained(
        model_name,  
        device_map='cuda:0',   
        quantization_config=quantization_config
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )


    return model, tokenizer


def get_activations_and_attention(
    model, tokenizer, text, start_layer, end_layer, token, max_token_division
):
    
    tokenizer.pad_token = tokenizer.eos_token
    if '12b' in model_name:
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
    else:
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        try:
            inputs = inputs.to("cuda:0")
            outputs = model.forward(inputs, output_hidden_states=True)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory. Printing memory status and attempting to clear cache.")
                for i in range(torch.cuda.device_count()):
                    print(f"Memory summary for GPU {i}:")
                    print(torch.cuda.memory_summary(device=i))
                torch.cuda.empty_cache()
            raise e

        last_tokens = []
        to_plot_activations = []

        for i in range(start_layer, end_layer):
            numpy_arr = outputs["hidden_states"][i][:, token].detach().cpu().numpy()
            mean = outputs["hidden_states"][i].mean(axis=1)

            last_tokens.append(mean.cpu())

        last_token_activations = torch.stack(last_tokens)
        
    return last_token_activations

def get_attention_weights(
    model, tokenizer, text, start_layer, end_layer
):
 
    inputs = tokenizer.encode(text, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    try:
        # Run model forward pass with attention weights enabled
        outputs = model(inputs, output_attentions=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Printing memory status and attempting to clear cache.")
            for i in range(torch.cuda.device_count()):
                print(f"Memory summary for GPU {i}:")
                print(torch.cuda.memory_summary(device=i))
            torch.cuda.empty_cache()
        raise e

    attention_weights = []
    for i in range(start_layer, end_layer):
        # Extract attention weights for the specified layer
        if i < len(outputs.attentions):
            layer_attention = outputs.attentions[i].detach().cpu()
            print(layer_attention)
            attention_weights.append(layer_attention)
        else:
            print(f"Layer {i} is out of range. Skipping.")

    return attention_weights


class Classifier(tf.keras.Model):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc1 = layers.Dense(
            768, kernel_initializer=initializers.HeUniform(), input_dim=input_dim
        )
        self.fc2 = layers.Dense(256, kernel_initializer=initializers.HeUniform())
        self.fc3 = layers.Dense(1)  
        self.dropout = layers.Dropout(0.5) 

    def call(self, x, training=False):
        x = self.fc1(x)
        x = tf.nn.relu(x) 
        x = self.dropout(x, training=training)  
        x = self.fc2(x)
        x = tf.nn.relu(x) 
        x = self.dropout(x, training=training) 
        x = self.fc3(x)
        x = tf.sigmoid(x)  
        return x


def build_classifier(input_dim):
    classifier = Classifier(input_dim)

    # Define the loss function
    criterion = losses.BinaryCrossentropy(from_logits=False)

    # Define the optimizer
    optimizer = optimizers.Adam()

    # Compile the model
    classifier.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])

    return classifier
