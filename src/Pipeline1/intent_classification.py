from transformers import pipeline

class AudioIntentClassification():

    def __init__(self, MODEL_NAME, filename):
        self.MODEL_NAME = MODEL_NAME
        self.filename = filename

    def audio_classification(self):
        classifier = pipeline(
            "audio-classification",
            model=self.MODEL_NAME
        )
        labels = classifier(
            self.filename,
            top_k=5
        )
        change = {
            'ang': 'anger',
            "hap": "happy",
            "sad": "sad",
            "neu": "neutral"
        }
        try:
            res = change[labels[0]['label']]
            return res
        except:
            pass
        return labels[0]['label']