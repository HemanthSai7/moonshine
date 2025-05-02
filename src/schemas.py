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

PredictInput = collections.namedtuple(
    "PredictInput",
    ("inputs", "inputs_length", "previous_tokens", "previous_encoder_states", "previous_decoder_states"),
    defaults=(None, None, None),
)

PredictOutput = collections.namedtuple(
    "PredictOutput",
    ("tokens", "next_tokens", "next_encoder_states", "next_decoder_states"),
    defaults = (None, None)
)

PredictOutputWithTranscript = collections.namedtuple(
    "PredictOutputWithTranscript",
    ("transcript", "tokens", "next_tokens", "next_encoder_states", "next_decoder_states"),
    defaults = (None, None)
)