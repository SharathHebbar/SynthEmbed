import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
from datasets import load_dataset

class TTS:
    def __init__(self, MODEL_NAME, HIFI_GAN_TTS, VOICES_DATASET, filename, text, sampling_rate):
        self.MODEL_NAME = MODEL_NAME
        self.HIFI_GAN_TTS = HIFI_GAN_TTS
        self.VOICES_DATASET = VOICES_DATASET
        self.filename = filename
        self.text = text
        self.sampling_rate = sampling_rate
        self.model = SpeechT5ForTextToSpeech.from_pretrainedd(self.MODEL_NAME)
        self.processor = SpeechT5Processor.from_pretrained(self.MODEL_NAME)
        self.vocoder = SpeechT5HifiGan.from_pretrained(HIFI_GAN_TTS)

    def tts(self):
        inputs = self.processor(
            text=self.text, return_tensors="pt"
            )
        
        embeddings_dataset = load_dataset(
            self.VOICES_DATASET, split="validation"
            )
        
        speaker_embeddings = torch.tensor(
            embeddings_dataset[30]["xvector"]
            ).unsqueeze(0)
        
        speech = self.model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=self.vocoder
            )
        
        sf.write(
            self.filename,
            speech.numpy(),
            samplerate=self.sampling_rate
            )
