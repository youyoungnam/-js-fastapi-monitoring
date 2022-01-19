import logging
import pandas as pd
from pathlib import Path
# 쉽게설명하자면, 파이썬내에서 pipeline을 경량화를 제공해주는 툴
# 이것을 사용하면 scale이나 모델을 파일로 저장할 수 있다.
from joblib import dump
from sklearn import preprocessing
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
def prepare_dataset(test_size=0.2, random_seed=1):
    dataset = pd.read_csv(
        "winequality-red.csv",
        delimiter=",",
    )
    dataset = dataset.rename(columns=lambda x: x.lower().replace(" ", "_"))
    train_df, test_df = train_test_split(dataset, test_size=test_size, random_state=random_seed)
    return {
        "train": train_df, 
        "test": test_df
        }
def train():
    logger.info("Preparing dataset...")
    dataset = prepare_dataset()
    train_df = dataset["train"]
    test_df = dataset["test"]

    # separate features from target
    y_train = train_df["quality"]
    X_train = train_df.drop("quality", axis=1)
    y_test = test_df["quality"]
    X_test = test_df.drop("quality", axis=1)

    logger.info("Training model...")
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(max_depth=30).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    logger.info(f"Test MSE: {error}")

    logger.info("Saving artifacts...")
    Path("artifacts").mkdir(exist_ok=True)
    # 모델과 scaler를 내보낸다.
    dump(model, "artifacts/model.joblib")
    dump(scaler, "artifacts/scaler.joblib")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()