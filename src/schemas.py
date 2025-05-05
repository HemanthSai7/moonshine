import collections

def TrainInput(inputs, inputs_length,):
    outputs = {
        "inputs": inputs,
        "inputs_length": inputs_length,
    }
    return outputs

def TrainOutput(logits, logits_length):
    outputs = {
        "logits": logits,
        "logits_length": logits_length,
    }
    return outputs

def TrainLabel(labels, labels_length):
    return {
        "labels": labels,
        "labels_length": labels_length,
    }