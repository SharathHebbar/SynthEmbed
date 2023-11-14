"""
UI for Audio Pipeline 1
"""

# Pre-defined Functions
import os
import wave
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
import pyaudio

# User Defined Functions
from file_manipulation import FileManipulation
from intent_classification import AudioIntentClassification
from stt import STT
from tts import TTS


class UI():
    """
    User Interface for Audio Pipeline 1
    """
    def __init__(self):
        load_dotenv()

        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL")
        self.search_api = os.getenv("SEARCH_API")
        self.stt_model = os.getenv("STT_MODEL")
        self.tts_model = os.getenv("TTS_MODEL")
        self.hifi_gan_tts = os.getenv("HIFI_GAN_TTS")
        self.voices_dataset = os.getenv("VOICES_DATASET")
        self.audio_classification_model = os.getenv("AUDIO_CLASSIFICATION_MODEL")
        self.audio_segmentation_model = os.getenv("AUDIO_SEGMENTATION_MODEL")
        self.text_classification_model = os.getenv("TEXT_CLASSIFICATION_MODEL")
        self.db_name = os.getenv("DB_NAME")
        self.sample_rate = os.getenv("SAMPLE_RATE")
        self.chunk = os.getenv("CHUNK")
        self.channels = os.getenv("CHANNELS")
        self.rate = os.getenv("RATE")
        self.record_seconds = os.getenv("RECORD_SECONDS")
        self.folder = os.getenv("AUDIO_DIR")
        self.format = pyaudio.paInt16

        self.user_id = uuid.uuid1()
        self.file_val = f"user_{self.user_id}"
        self.filename = None
        self.file_upload_status = None
        self.folder_path = os.path.join(self.folder, self.file_val)

        # file_manipulation_obj = FileManipulation(self.folder)
        # file_manipulation_obj.make_directory_if_not_exists()

        self.radio_label = "Record"
        self.record_disability = True
        self.uploader_visibility = False

        self.audio_intent = None
        self.transcription = None
        
        self.text = None
        self.filename_tts = None

    def record(self):
        """
        This function records user audio
        """
        # self.filename = f"{self.file_val}/user_{self.user_id}.wav"
        self.filename = f"audios/user_{self.user_id}.wav"
        with open(self.filename, "wb") as audio_file:
            pyaudio_obj = pyaudio.PyAudio()
            audio_file.setnchannels(self.channels)
            audio_file.setsamplewidth(pyaudio_obj.get_sample_size(self.format))
            audio_file.setframerate(self.rate)

            stream = pyaudio_obj.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True)

            for _ in range(0, self.rate // self.chunk * self.record_seconds):
                audio_file.writeframes(stream.read(self.chunk))
        stream.close()
        pyaudio_obj.terminate()


    def radio_button(self):
        """
        Function used to create radio button
        """
        options = ['Upload', 'Record']
        radio_label = st.radio(label="Choose your option", options=options)
        return radio_label


    def upload_audio(self):
        """
        Function used to upload audio
        """
        if self.radio_label == "Upload":
            self.uploader_visibility = False
        else:
            self.uploader_visibility = True

        self.file_upload_status = st.file_uploader(
            label="Upload a Audio File (preferably .wav)",
            disabled=self.uploader_visibility
        )

        if self.file_upload_status is not None:
            # self.filename = f"{self.file_val}/user_{self.user_id}.wav"
            self.filename = f"audios/user_{self.user_id}.wav"
            with open(self.filename, "wb") as file_obj:
                file_obj.write(self.file_upload_status.getbuffer())


    def main(self):
        """
        Main function
        """
        st.set_page_config(
        page_title="Audio Pipeline",
        layout="wide",
        page_icon="🔉"
        )

        st.title("Audio Pipeline")
        try:
            if not self.huggingface_api_key:
                st.warning("Please Enter your API Key")
            # self.radio_label = self.radio_button()

            # if self.radio_label == "Record":
            #     self.record_disability = False
            # elif self.radio_label == "Upload":
            #     self.record_disability = True

            # # file_manipulation_obj = FileManipulation(self.folder_path)
            # # file_manipulation_obj.make_directory_if_not_exists()

            # self.upload_audio()
            
            st.button(
                label="Record🎙️",
                on_click=self.record(),
                disabled=self.record_disability
            )

            time.sleep(self.record_seconds)
            return True
        except Exception as exp:
            st.error(exp, icon="🚨")
            return False
    
    def pipeline(self):
        """
        Pipeline1
        """

        # Intent Classification for audio

        audio_classification_obj = AudioIntentClassification(self.audio_classification_model, self.filename)
        self.audio_intent = audio_classification_obj.audio_classification()

        # Audio Segmentation
        # There are no auto model for segmentation

        # Speech to text

        stt_obj = STT(self.stt_model, self.filename, self.sample_rate)
        self.transcription = stt_obj.stt()


        # All of text related tasks
        # Text to speech
        # self.filename_tts = f"{self.file_val}/tts_user_{self.user_id}.wav"
        self.filename_tts = f"audios/tts_user_{self.user_id}.wav"
        tts_obj = TTS(
            self.tts_model,
            self.hifi_gan_tts,
            self.voices_dataset,
            self.filename_tts,
            self.text,
            self.sample_rate
            )
        
        tts_obj.tts()
        return self.filename_tts



ui = UI()
ui.main()