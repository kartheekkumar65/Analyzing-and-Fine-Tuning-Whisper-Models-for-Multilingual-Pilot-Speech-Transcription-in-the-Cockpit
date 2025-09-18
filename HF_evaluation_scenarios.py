from datasets import DatasetDict, Dataset, Audio
import os
import pandas as pd
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process audio and generate transcripts")
    parser.add_argument("--model_name", type=str, default="medium", help="HF Model")

    return parser.parse_args()

args = parse_arguments()
model_name = args.model_name

# Paths to audio and transcript folders
audio_folder = f"./All_Segments/"  
transcript_folder = f"./All_Segments/" 

# Step 1: Load the audio and transcript file names
def load_data(audio_folder, transcript_folder):
    data = []
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith(".mp3"):  # Assuming audio files are in .wav format
            audio_path = os.path.join(audio_folder, audio_file)
            transcript_path = os.path.join(transcript_folder, audio_file.replace(".mp3", ".txt"))
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                data.append({"audio": audio_path, "transcript": transcript})
    return data

# Step 2: Create a DataFrame
data = load_data(audio_folder, transcript_folder)
df = pd.DataFrame(data)

# Step 3: Convert the DataFrame into a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Step 4: Cast the "audio" column to the Audio feature type
dataset = dataset.cast_column("audio", Audio())

# Step 5: Wrap in a DatasetDict
dataset_dict = DatasetDict({"train": dataset})  # Replace "train" with other splits if needed

# Step 6: Verify the DatasetDict
print(dataset_dict)
print(dataset_dict["train"][0])  # Check the first data sample

# Print the sizes of the train and test splits
print(f"Test set size: {len(dataset_dict['train'])}")

from datasets import Audio
dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))

model_name_or_path = f"openai/whisper-{args.model_name}"; task = "transcribe"; language = 'de'

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, task=task, language = language)
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(model_name_or_path, task=task, language = language)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch


dataset_dict = dataset_dict.map(prepare_dataset, remove_columns=dataset_dict.column_names["train"], num_proc=1)


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
import evaluate
metric = evaluate.load("wer")

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto")

import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
# from helper import normalize_text as proposed_normalizer
# from whisper.normalizers import EnglishTextNormalizer 

eval_dataloader = DataLoader(dataset_dict["train"], batch_size=4, collate_fn=data_collator)
forced_decoder_ids = processor.get_decoder_prompt_ids( task=task, language = language)
basic_normalizer = BasicTextNormalizer()
# opAI_normalizer = EnglishTextNormalizer()
from helper import proposed_1, proposed_2, proposed_3, number_normalize, whisper_normalize

predictions = []
references = []
basic_normalized_predictions = []
basic_normalized_references = []
number_normalized_predictions = []
number_normalized_references = []
opAI_normalized_predictions = []
opAI_normalized_references = []
proposed_1_normalized_predictions = []
proposed_1_normalized_references = []
proposed_2_normalized_predictions = []
proposed_2_normalized_references = []
proposed_3_normalized_predictions = []
proposed_3_normalized_references = []

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            basic_normalized_predictions.extend([basic_normalizer(pred).strip() for pred in decoded_preds])
            basic_normalized_references.extend([basic_normalizer(label).strip() for label in decoded_labels])
            number_normalized_predictions.extend([number_normalize(pred).strip() for pred in decoded_preds])
            number_normalized_references.extend([number_normalize(label).strip() for label in decoded_labels])
            opAI_normalized_predictions.extend([whisper_normalize(pred).strip() for pred in decoded_preds])
            opAI_normalized_references.extend([whisper_normalize(label).strip() for label in decoded_labels])
            proposed_1_normalized_predictions.extend([proposed_1(pred).strip() for pred in decoded_preds])
            proposed_1_normalized_references.extend([proposed_1(label).strip() for label in decoded_labels])
            proposed_2_normalized_predictions.extend([proposed_2(pred).strip() for pred in decoded_preds])
            proposed_2_normalized_references.extend([proposed_2(label).strip() for label in decoded_labels])
            proposed_3_normalized_predictions.extend([proposed_3(pred).strip() for pred in decoded_preds])
            proposed_3_normalized_references.extend([proposed_3(label).strip() for label in decoded_labels])

        del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric.compute(predictions=predictions, references=references)
basic_normalized_wer = 100 * metric.compute(predictions=basic_normalized_predictions, references=basic_normalized_references)
number_normalized_wer = 100 * metric.compute(predictions=number_normalized_predictions, references=number_normalized_references)
opAI_normalized_wer = 100 * metric.compute(predictions=opAI_normalized_predictions, references=opAI_normalized_references)
proposed_1_normalized_wer = 100 * metric.compute(predictions=proposed_1_normalized_predictions, references=proposed_1_normalized_references)
proposed_2_normalized_wer = 100 * metric.compute(predictions=proposed_2_normalized_predictions, references=proposed_2_normalized_references)
proposed_3_normalized_wer = 100 * metric.compute(predictions=proposed_3_normalized_predictions, references=proposed_3_normalized_references)


# Save the results into a text file
f = open(f"./HF_{model_name}_metrics.txt", "w")
f.write(f"wer = {wer:.4f}\n")
f.write(f"Basic Normalized wer = {basic_normalized_wer:.4f}\n")
f.write(f"Number Normalized wer = {number_normalized_wer:.4f}\n")
f.write(f"OpAI Normalized wer = {opAI_normalized_wer:.4f}\n")
f.write(f"Proposed 1 Normalized wer = {proposed_1_normalized_wer:.4f}\n")
f.write(f"Proposed 2 Normalized wer = {proposed_2_normalized_wer:.4f}\n")
f.write(f"Proposed 3 Normalized wer = {proposed_3_normalized_wer:.4f}\n")
f.close()

print(f"wer = {wer:.4f}")
print(f"Basic Normalized wer = {basic_normalized_wer:.4f}")
print(f"Number Normalized wer = {number_normalized_wer:.4f}")
print(f"OpAI Normalized wer = {opAI_normalized_wer:.4f}")
print(f"Proposed 1 Normalized wer = {proposed_1_normalized_wer:.4f}")
print(f"Proposed 2 Normalized wer = {proposed_2_normalized_wer:.4f}")
print(f"Proposed 3 Normalized wer = {proposed_3_normalized_wer:.4f}")