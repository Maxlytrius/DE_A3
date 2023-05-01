from ray import tune
import ray
import os
import numpy as np

NUM_MODELS = 100

def train_model(config):
    score = config["model_id"]

    # Import model libraries, etc...
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import fetch_covtype
    # Load data and train model code here...
    X, y = fetch_covtype(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(X,y)
    # Return final stats. You can also return intermediate progress
    # using ray.air.session.report() if needed.
    # To return your model, you could write it to storage and return its
    # URI in this dict, or return it as a Tune Checkpoint:
    # https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
    return {"score": score, "other_data": ...}


if __name__ == '__main__':
    # Define trial parameters as a single grid sweep.
    trial_space = {
        # This is an example parameter. You could replace it with filesystem paths,
        # model types, or even full nested Python dicts of model configurations, etc.,
        # that enumerate the set of trials to run.
        my_list = np.arange(start, stop + step, step)
}