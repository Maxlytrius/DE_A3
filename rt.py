from ray import tune
import ray
import os
import numpy as np

NUM_MODELS = 100

def train_model(config):

    # Import model libraries, etc...
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import cross_val_score
    # Load data and train model code here...
    X, y = fetch_covtype(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"], ccp_alpha=config["ccp_alpha"])
    clf.fit(X,y)
    score = cross_val_score(clf, X, y, cv=5)
    # Return final stats. You can also return intermediate progress
    # using ray.air.session.report() if needed.
    # To return your model, you could write it to storage and return its
    # URI in this dict, or return it as a Tune Checkpoint:
    # https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
    return {"score": score, "other_data": "buh"}


if __name__ == '__main__':
    # Define trial parameters as a single grid sweep.
    trial_space = {
        # This is an example parameter. You could replace it with filesystem paths,
        # model types, or even full nested Python dicts of model configurations, etc.,
        # that enumerate the set of trials to run.
        "max_depth": tune.grid_search(np.arange(0, 20, 1)) 
        "n_estimators": tune.grid_search(np.arange(0, 20, 1)) 
        "ccp_alpha" : tune.grid_search(np.arange(0.00, 0.5, 0.01)) 
    }

    train__with_resources = tune.with_resources(train_model, {"cpu": 1})
    tuner = tune.Tuner(
    train__with_resources,
    tune_config=trial_space
    )
    results = tuner.fit()