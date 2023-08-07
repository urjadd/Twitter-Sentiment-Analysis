import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commons.generic_constants import GenericConstants
from scripts.training_model_clean_data import TrainingModelCleanData
from scripts.train_sentiment_model import TrainSentimentModel


class ExecuteProject:

    def __init__(self, run_type, clean_data=None):
        self.run_type = run_type
        self.clean_data = clean_data

    def run_sentiment_analysis(self):
        if self.run_type == GenericConstants.TRAINING and self.clean_data:
            clean_data = TrainingModelCleanData()
            clean_data.clean_twitter_data()

        elif self.run_type == GenericConstants.TRAINING and not self.clean_data:
            sentiment_model = TrainSentimentModel()
            sentiment_model.train_sentiment_model()


if __name__ == '__main__':
    cmd_line_args = sys.argv
    clean = False
    if len(cmd_line_args) > 2:
        if cmd_line_args[1:][1].lower() == 'clean':
            clean = True
    execute_project_obj = ExecuteProject(cmd_line_args[1:][0], clean)
    execute_project_obj.run_sentiment_analysis()
