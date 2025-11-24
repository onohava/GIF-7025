from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, model):
        self.model = model