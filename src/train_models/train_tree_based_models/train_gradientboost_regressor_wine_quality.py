from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
from src.data.make_wine_quality_dataset import WineQuality
from scipy.stats import randint, uniform


def training():
    train_set = WineQuality(
        train=True, in_folder='data/raw', out_folder='data/processed')

    X_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()

    quantiles = [0.05, 0.1, 0.2, 0.8, 0.9, 0.95]
    for quantile in quantiles:
        print(f"start training model for q{quantile}")
        params_distributions = dict(
            max_leaf_nodes=randint(low=10, high=50),
            max_depth=randint(low=3, high=20),
            n_estimators=randint(low=50, high=300),
            learning_rate=uniform()
        )
        qr = GradientBoostingRegressor(alpha=quantile, loss='quantile')
        model = RandomizedSearchCV(qr, params_distributions)
        model.fit(X_train, y_train)
        with open(f'models/trained_gradientboost_regressor{quantile}_wine_quality.pkl', 'wb') as f:
            pickle.dump(model, f)
    print("finished training")


if __name__ == "__main__":
    training()
