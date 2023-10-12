#!/usr/bin/env python3
import sys
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import pandas as pd
import numpy as np
from utils import convert_class_name_to_id, display_all_results_dnn


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print('Usage: python evaluation.py <evaluation_set.csv>')
        sys.exit(1)

    # Load the tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("../models")

    # Load the model
    model = CamembertForSequenceClassification.from_pretrained("../models")

    # Tokenizing the verbatim
    evaluation_set_path = sys.argv[1]
    evaluation_data = pd.read_csv(evaluation_set_path)

    data_texts = evaluation_data['text'].to_list()
    X_test = tokenizer(data_texts, truncation=True, padding=True, return_tensors='pt')

    data_labels = evaluation_data.label.copy()
    y_test = convert_class_name_to_id(data_labels, model.config.label2id)

    # Inference
    with torch.no_grad():
        results = model(**X_test).logits.numpy()

    # Model Evaluation
    y_pred = [np.argmax(pred) for pred in results]

    class_names = list(model.config.id2label.values())
    display_all_results_dnn(y_test, y_pred, class_names)


if __name__ == "__main__":
    main()