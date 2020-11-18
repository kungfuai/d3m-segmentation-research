import pandas as pd


class PredictionReport:
    def __init__(self):
        self.predictions = []

    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def print_predictions(self):
        print(self.to_dataframe())

    def save_predictions(self, path):
        self.to_dataframe().to_csv(path, header=True, index=False)

    def to_dataframe(self):
        df = pd.DataFrame()
        return df
