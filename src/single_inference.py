#!/usr/bin/env python3
import sys
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import numpy as np
import time


def main():
    """
    This function load the pretrained classification model and performs a single inference on the given verbatim
    Args:
    Return:
    """
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print('Usage: python single_inference.py "<verbatim>"')
        sys.exit(1)

    # Load the tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("../models")

    # Load the model
    model = CamembertForSequenceClassification.from_pretrained("../models")

    # Start the inference and mesuring the time
    start_time = time.time()

    # Tokenizing the verbatim
    test_verbatim = sys.argv[1]
    encoded_test = tokenizer(test_verbatim, return_tensors='pt')

    # Inference
    with torch.no_grad():
        results = model(**encoded_test).logits

    print(model.config.id2label[int(np.argmax(results))])

    inference_time = time.time() - start_time
    print(f"Inference time : {inference_time} s")


if __name__ == "__main__":
    main()