import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataCleaningPipeline:
    def __init__(self, config=None):
        """
        config: dict containing cleaning strategies
        """
        self.config = config if config else {}
        self.scaler = None
        logging.info("Pipeline initialized with config: %s", self.config)

    def handle_missing_values(self, df, strategy="auto"):
        try:
            if strategy == "drop":
                df = df.dropna()
                logging.info("Dropped missing values")
            elif strategy == "mean":
                df = df.fillna(df.mean(numeric_only=True))
                logging.info("Filled missing values with mean")
            elif strategy == "median":
                df = df.fillna(df.median(numeric_only=True))
                logging.info("Filled missing values with median")
            elif strategy == "mode":
                df = df.fillna(df.mode().iloc[0])
                logging.info("Filled missing values with mode")
            else:  # auto: forward fill
                df = df.ffill().bfill()
                logging.info("Applied forward/backward fill for missing values")
            return df
        except Exception as e:
            logging.error("Error handling missing values: %s", e)
            raise

    def handle_outliers(self, df, method="iqr"):
        try:
            if method == "iqr":
                for col in df.select_dtypes(include=[np.number]).columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    df[col] = np.where(df[col] < lower, lower,
                                       np.where(df[col] > upper, upper, df[col]))
                logging.info("Outliers handled using IQR method")
            return df
        except Exception as e:
            logging.error("Error handling outliers: %s", e)
            raise

    def encode_categorical(self, df, encoding_map=None):
        try:
            if encoding_map:
                for col, method in encoding_map.items():
                    if method == "onehot":
                        df = pd.get_dummies(df, columns=[col], drop_first=True)
                        logging.info("Applied one-hot encoding to %s", col)
                    elif method == "label":
                        df[col] = df[col].astype("category").cat.codes
                        logging.info("Applied label encoding to %s", col)
            return df
        except Exception as e:
            logging.error("Error encoding categorical variables: %s", e)
            raise

    def scale_features(self, df, scaler_type="standard"):
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            elif scaler_type == "robust":
                self.scaler = RobustScaler()
            elif scaler_type == "maxabs":
                self.scaler = MaxAbsScaler()
            else:
                raise ValueError("Invalid scaler type")

            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logging.info("Applied %s scaling", scaler_type)
            return df
        except Exception as e:
            logging.error("Error scaling features: %s", e)
            raise

    def engineer_features(self, df):
        try:
            if "price" in df.columns and "quantity" in df.columns:
                df["total_value"] = df["price"] * df["quantity"]
                logging.info("Engineered feature: total_value")
            return df
        except Exception as e:
            logging.error("Error engineering features: %s", e)
            raise

    def fit(self, df):
        try:
            # Learn parameters (e.g., scaler fit)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.scaler = StandardScaler().fit(df[numeric_cols])
            logging.info("Fitted scaler on training data")
        except Exception as e:
            logging.error("Error fitting pipeline: %s", e)
            raise

    def transform(self, df):
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if self.scaler:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
                logging.info("Transformed dataset using fitted scaler")
            return df
        except Exception as e:
            logging.error("Error transforming dataset: %s", e)
            raise

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save_pipeline(self, filepath):
        try:
            joblib.dump(self, filepath)
            logging.info("Pipeline saved to %s", filepath)
        except Exception as e:
            logging.error("Error saving pipeline: %s", e)
            raise

    def load_pipeline(self, filepath):
        try:
            pipeline = joblib.load(filepath)
            logging.info("Pipeline loaded from %s", filepath)
            return pipeline
        except Exception as e:
            logging.error("Error loading pipeline: %s", e)
            raise
pipeline = DataCleaningPipeline()

# Simulate dataset
df = pd.DataFrame({
    "price":[10,20,30,np.nan,1000],
    "quantity":[1,2,3,4,5],
    "category":["A","B","A","C","B"]
})

df = pipeline.handle_missing_values(df, strategy="mean")
df = pipeline.handle_outliers(df, method="iqr")
df = pipeline.encode_categorical(df, {"category":"label"})
df = pipeline.scale_features(df, scaler_type="standard")
df = pipeline.engineer_features(df)

print(df.head())     


#part B
import pandas as pd
import numpy as np

# Simulate Orders table
orders_df = pd.DataFrame({
    "order_id": np.arange(1, 10001),
    "user_id": np.random.randint(1, 1000, size=10000),
    "amount": np.random.uniform(10, 500, size=10000),
    "status": np.random.choice(["Completed","Pending","Cancelled"], size=10000)
})

print(orders_df.head())