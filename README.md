# Analyzing-and-Fine-Tuning-Whisper-Models-for-Multilingual-Pilot-Speech-Transcription-in-the-Cockpit

This repository contains software mentioned in the ICASSP 2026 submission "Analyzing and Fine-Tuning Whisper Models for Multilingual Pilot Speech Transcription in the Cockpit."

### Install the required packages
```pip install -U requirements.txt```

### Download the English Dictionary
``` python -m spacy download en_core_web_sm ```

### Add the audio and transcript files to All_Segments folder
- audio files should conform to .mp3 format with max length < 30 seconds
- transcript files (.txt) should be of same name as corresponding audio file  
- Ex: audio file: speech_1.mp3, transcript file: speech_1.txt

### Run Transcription and Evaluation
``` python HF_evaluation_scenarios.py --model_name medium |& tee HF_medium_eval.txt```

### Run Finetuning
```python HF_finetuning_LoRA.py --model_name medium --learning_rate 1e-3 |& tee HF_finetune_medium_1e-3.txt```

