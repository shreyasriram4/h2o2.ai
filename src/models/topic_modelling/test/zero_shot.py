from src.utils.file_util import FileUtil
from transformers import pipeline

class ZeroShot():
    def save_model(self):
        classifier = pipeline(task="zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device=-1)
        FileUtil.put_topic_model(classifier)
    
    def get_model(self):
        return FileUtil.get_topic_model()
    
if __name__ == "__main__":
    ZeroShot().save_model()