from preprocessing.stopwords import STOP_WORDS
import string

class Cleaner:
    def __init__(self):
        self.translator = str.maketrans('', '', string.punctuation)


    def lowercase(self,text):
        return text.lower()


    def remove_punctuation(self,text):
        return text.translate(self.translator)


    def remove_stopwords(self,text):
        data = []
        for word in text.split():
            if word not in STOP_WORDS:
                data.append(word)

        return " ".join(data)


    def clean_text(self,text):
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)


        return text


    def clean(self,data):
        texts = []
        for text in data:
            texts.append(self.clean_text(text))

        return texts