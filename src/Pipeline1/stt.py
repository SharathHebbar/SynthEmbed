import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf

class STT:

    def __init__(self, MODEL_NAME, filename, sampling_rate):
        self.MODEL_NAME = MODEL_NAME
        self.filename = filename
        self.sampling_rate = sampling_rate
        self.model = Speech2TextForConditionalGeneration.from_pretrained(self.MODEL_NAME)
        self.processor = Speech2TextProcessor.from_pretrained(self.MODEL_NAME)

    def map_audio_to_array(self):
        self.speech_array, _ = sf.read(self.filename)

    def stt(self):
        input_features = self.processor(
            self.speech_array,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features
        ids = self.model(
            input_features
        )
        transcription = self.processor.batch_decode(
            ids
        )
        return transcription