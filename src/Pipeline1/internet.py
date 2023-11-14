import requests
import json


class Brain2:
    def __init__(self, query: str, URL: str, SEARCH_API_KEY: str):
        self.query = query
        self.URL = URL
        self.SEARCH_API_KEY = SEARCH_API_KEY

        self.headers = {
            "X-API-KEY": self.SEARCH_API_KEY,
            "Content-Type": 'application/json' # can be a input parameter
        }

    def search_results(self):
        payload = json.dumps(
            {
                "q": self.query,
                "gl": "in" # can be given as input
            }
        )

        response = requests.request(
            "POST",
            self.URL,
            headers=self.headers,
            data=payload
        )

        result = json.load(response)
        return result['answerBox']['answer']
