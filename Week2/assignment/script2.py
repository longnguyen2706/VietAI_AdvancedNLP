from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
import os
from transformers import EncoderDecoderModel, RobertaTokenizer, RobertaForMaskedLM

"""# Prepare data processing function"""

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from dataclasses import dataclass
from transformers.utils import PaddingStrategy
from typing import Optional, Union, Any
import numpy as np
from datasets import load_metric
import torch
from datasets import load_dataset

from datasets import Dataset

cache_dir = './cache'
model_name = 'nguyenvulebinh/envibert'
def download_tokenizer_files():
    resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
    for item in resources:
        if not os.path.exists(os.path.join(cache_dir, item)):
            tmp_file = hf_bucket_url(model_name, filename=item)
            tmp_file = cached_path(tmp_file, cache_dir=cache_dir)
            os.rename(tmp_file, os.path.join(cache_dir, item))

def init_tokenizer():
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    return tokenizer
def init_model():
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    # set encoder decoder tying to True
    roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name,
                                                                         model_name,
                                                                         tie_encoder_decoder=False)

    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    roberta_shared.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    # set decoding params
    roberta_shared.config.max_length = 100
    roberta_shared.config.early_stopping = True
    roberta_shared.config.no_repeat_ngram_size = 3
    roberta_shared.config.length_penalty = 2.0
    roberta_shared.config.num_beams = 1
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    return roberta_shared, tokenizer

def get_dataset():
    dataset = load_dataset('VietAI/spoken_norm_assignment')
    # map_column(dataset['train'])
    # map_column(dataset['test'])
    # map_column(dataset['valid'])
    dataset = dataset.rename_column("src", "input_ids")
    dataset = dataset.rename_column("tgt", "labels")
    return dataset

# def map_column(data):
#     data['input_ids'] = data['src']
#     data['labels'] = data['tgt']
#     del data['src']
#     del data['tgt']

def flatten(data):
    inputs = []
    labels = []
    for row in data:
        inputs+= row['input_ids']
        labels+= row['labels']
    # print (len(inputs))
    # print (len(labels))
    return {'input_ids': inputs, 'labels': labels}

def flatten2(data):
    input_ids = list(np.concatenate(data['input_ids']))
    labels =  list(np.concatenate(data['labels']))
    return {'input_ids': input_ids, 'labels': labels}

def get_metric_compute_fn(tokenizer):
    metric = load_metric('sacrebleu')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        print (result)
        return {"bleu": result["score"]}

    return compute_metrics

@dataclass
class DataCollatorForEnViMT:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        features_tokenized = []
        for feature in features:
            src = feature['input_ids']
            tgt = feature['labels']
            if len(src) == 0 or len(tgt) == 0:
                continue
            features_tokenized.append({'input_ids': tokenizer(src)["input_ids"],
                                       'labels': tokenizer(tgt)["input_ids"][1:]})

        features = features_tokenized
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))

                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

model, tokenizer = init_model()
# print(model)

data = get_dataset()
print (data)

# flatten_data = data
# flatten_data['train'] = data['train'].flatten()
# print (flatten_data)
# train_set = Dataflatten(data['train'])

# train_dict = {'input_ids': [], 'labels': []}
# for i in range (0, 500000//2500):
#     dict = flatten2(data['train'][i:(i+1)*2500])
#     train_dict['input_ids'] += dict['input_ids']
#     train_dict['labels'] += dict['labels']
                    
train_set = Dataset.from_dict(flatten2(data['train'][0:2500]))
val_set = Dataset.from_dict(flatten(data['valid']))
print(train_set, val_set)

# # Metrics
# wer = load_metric('wer')
# predictions = [' '.join(item) for item in data['valid']['src']]
# references = [' '.join(item) for item in data['valid']['tgt']]
# wer_score = wer.compute(predictions=predictions,
#             references=references)
# print("wer score: ", wer_score)


data_collator = DataCollatorForEnViMT(tokenizer, model=model)
num_epochs = 1
checkpoint_path = "./envi_checkpoints"
batch_size = 16  # change to 16 for full training
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_path,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=1,
    predict_with_generate=True,
    save_total_limit=2,
    do_train=True,
    do_eval=True,
    logging_steps=10,
    num_train_epochs = num_epochs,
    warmup_ratio=1 / num_epochs,
    logging_dir=os.path.join(checkpoint_path, 'log'),
    overwrite_output_dir=True,
    metric_for_best_model='bleu',
    greater_is_better=True,
    eval_accumulation_steps=10,
    dataloader_num_workers=12
    # sharded_ddp="simple",
    # fp16=True,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=get_metric_compute_fn(tokenizer),
    train_dataset=train_set ,  # Only use subset of the dataset for a quick training. Remove shard for full training
    eval_dataset=val_set, # Only use subset of the dataset for a quick training. Remove shard for full training
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# trained_model = model.from_pretrained('./envi_checkpoints/checkpoint-9020')
#
# """## Decoding with beam"""
#
# # encode context the generation is conditioned on
# input_ids = tokenizer.encode('I enjoy walking with my girl', return_tensors='pt')
#
# # generate text until the output length (which includes the context length) reaches 20
# beam_outputs = trained_model.generate(
#     input_ids,
#     max_length=20,
#     num_beams=10,
#     no_repeat_ngram_size=2,
#     num_return_sequences=5,
#     early_stopping=True
# )
#
# print("Output:\n" + 100 * '-')
# for i, beam_output in enumerate(beam_outputs):
#   output_pieces = tokenizer.convert_ids_to_tokens(beam_output.numpy().tolist())
#   output_text = tokenizer.sp_model.decode(output_pieces)
#   print("{}: {}".format(i, output_text))