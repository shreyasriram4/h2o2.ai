import os
import pandas as pd
from src.models.classifier import Classifier
from src.utils.file_util import FileUtil
from transformers import pipeline

class ZeroShot(Classifier):
    def save_model(self):
        classifier = pipeline(task="zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device=-1)
        FileUtil.put_topic_model(classifier)
    
    def get_model(self):
        model_file_path = os.path.join(FileUtil.TOPIC_MODELLING_DIR, FileUtil.MODEL_FILE_NAME)
        if not FileUtil.check_filepath_exists(model_file_path):
            self.save_model()
        return FileUtil.get_topic_model()
    
    def dataloader(self, df, column):
        for i in range(len(df)):
            yield df.loc[i, column]
    
    def predict(self, df, column, candidate_labels):
        clf = self.get_model()
        hypothesis_template = "The topic of this review is {}."

        preds = clf(
            self.dataloader(df, column),
            candidate_labels, 
            hypothesis_template = hypothesis_template
            )

        preds = pd.DataFrame(preds)
        df['topic'] = preds['labels'].apply(lambda x: x[0])

        return df
    
    def fit(self):
        pass

    def evaluate(self):
        pass
    
if __name__ == "__main__":
    ZeroShot().save_model()