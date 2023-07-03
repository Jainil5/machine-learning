from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import_model = "t5-small"
tokenizer  = AutoTokenizer.from_pretrained(import_model)
model =AutoModelForSequenceClassification.from_pretrained(import_model)

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

print(preprocess_function("Hello i am fine."))