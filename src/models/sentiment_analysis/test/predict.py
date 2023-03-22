# pending until BERT training is done



# import os
# import pandas as pd
# import yaml
# from src.models.topic_modelling.test.zero_shot import ZeroShot

# PROJECT_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
# )
# SENTIMENT_ANALYSIS_DIR = os.path.join(PROJECT_DIR, "src/models/sentiment_analysis")
# SENTIMENT_ANALYSIS_MODEL_FILE_NAME = "model.pkl"

# def predict_topic(df):
#     with open(os.path.join(SENTIMENT_ANALYSIS_DIR, SENTIMENT_ANALYSIS_MODEL_FILE_NAME), 'r') as file:
#         clf = pickle.load(f)

#     clf = ZeroShot().get_model()

#     hypothesis_template = "The topic of this review is {}."

#     single_topic_prediction = clf(
#         list(df["cleaned_text"]), 
#         candidate_labels, 
#         hypothesis_template = hypothesis_template
#         )

#     single_topic_prediction = pd.DataFrame(single_topic_prediction)

#     df['topic'] = single_topic_prediction['labels'].apply(lambda x: x[0])

#     return df

# if __name__ == "__main__":
#     predict_topic()