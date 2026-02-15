"""
ML Anomaly Detection Model
Uses Isolation Forest for unsupervised fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yaml
from typing import Dict, List, Tuple
import joblib
from datetime import datetime

class AnomalyDetector:
    """ML-based anomaly detection for merchant fraud"""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        model_config = self.config['models']['anomaly_detection']
        self.model = IsolationForest(
            n_estimators=model_config['n_estimators'],
            contamination=model_config['contamination'],
            max_samples=model_config['max_samples'],
            random_state=model_config['random_state'],
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = None

    def engineer_features(self, merchants_df: pd.DataFrame, 
                         transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for anomaly detection"""
        print("Engineering features...")

        features = merchants_df.copy()

        # Transaction-based features
        txn_features = self._calculate_transaction_features(transactions_df)
        features = features.merge(txn_features, on='merchant_id', how='left')

        # Fill NaN for merchants with no transactions
        features = features.fillna(0)

        # Select final feature set
        self.feature_columns = [
            'industry_risk_weight',
            'business_age_days',
            'monthly_revenue',
            'chargeback_rate',
            'has_website',
            'has_social_media',
            'txn_count',
            'avg_transaction_amount',
            'std_transaction_amount',
            'max_transaction_amount',
            'velocity_1h_max',
            'velocity_24h_count',
            'unique_customers',
            'repeat_customer_rate',
            'card_present_ratio',
            'night_transaction_ratio'
        ]

        return features

    def _calculate_transaction_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction-based features"""

        features = []

        for merchant_id in transactions_df['merchant_id'].unique():
            merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id]

            # Basic counts
            txn_count = len(merchant_txns)

            # Amount statistics
            amounts = merchant_txns['amount']
            avg_amount = amounts.mean()
            std_amount = amounts.std()
            max_amount = amounts.max()

            # Velocity features
            merchant_txns = merchant_txns.sort_values('timestamp')
            merchant_txns['hour'] = merchant_txns['timestamp'].dt.floor('H')
            hourly_counts = merchant_txns.groupby('hour').size()
            velocity_1h_max = hourly_counts.max() if len(hourly_counts) > 0 else 0

            # 24-hour velocity
            recent_24h = merchant_txns[
                merchant_txns['timestamp'] > merchant_txns['timestamp'].max() - pd.Timedelta(hours=24)
            ]
            velocity_24h_count = len(recent_24h)

            # Customer features
            unique_customers = merchant_txns['customer_id'].nunique()
            customer_counts = merchant_txns['customer_id'].value_counts()
            repeat_customers = (customer_counts > 1).sum()
            repeat_customer_rate = repeat_customers / unique_customers if unique_customers > 0 else 0

            # Payment method features
            card_present_ratio = merchant_txns['card_present'].mean()

            # Temporal features
            merchant_txns['hour_of_day'] = merchant_txns['timestamp'].dt.hour
            night_txns = ((merchant_txns['hour_of_day'] < 6) | 
                         (merchant_txns['hour_of_day'] > 23)).sum()
            night_transaction_ratio = night_txns / len(merchant_txns) if len(merchant_txns) > 0 else 0

            features.append({
                'merchant_id': merchant_id,
                'txn_count': txn_count,
                'avg_transaction_amount': avg_amount,
                'std_transaction_amount': std_amount if not pd.isna(std_amount) else 0,
                'max_transaction_amount': max_amount,
                'velocity_1h_max': velocity_1h_max,
                'velocity_24h_count': velocity_24h_count,
                'unique_customers': unique_customers,
                'repeat_customer_rate': repeat_customer_rate,
                'card_present_ratio': card_present_ratio,
                'night_transaction_ratio': night_transaction_ratio
            })

        return pd.DataFrame(features)

    def fit(self, features_df: pd.DataFrame) -> 'AnomalyDetector':
        """Train the anomaly detection model"""
        print("Training Isolation Forest model...")

        # Prepare features
        X = features_df[self.feature_columns].copy()

        # Convert boolean to int
        X['has_website'] = X['has_website'].astype(int)
        X['has_social_media'] = X['has_social_media'].astype(int)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled)

        print("Model training complete")
        return self

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomaly scores"""
        # Prepare features
        X = features_df[self.feature_columns].copy()
        X['has_website'] = X['has_website'].astype(int)
        X['has_social_media'] = X['has_social_media'].astype(int)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        # anomaly_score: -1 for anomalies, 1 for normal
        # Convert to 0-100 risk score (higher = more anomalous)
        raw_scores = self.model.decision_function(X_scaled)

        # Convert to 0-100 scale (higher = more risky)
        # raw_scores range: [-0.5, 0.5] approximately
        anomaly_scores = ((0.5 - raw_scores) * 100).clip(0, 100)

        predictions = features_df[['merchant_id']].copy()
        predictions['ml_anomaly_score'] = anomaly_scores
        predictions['is_anomaly'] = self.model.predict(X_scaled) == -1

        return predictions

    def evaluate(self, features_df: pd.DataFrame, ground_truth: pd.Series) -> Dict:
        """Evaluate model performance"""
        predictions = self.predict(features_df)

        y_true = ground_truth.values
        y_pred = predictions['is_anomaly'].values

        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'auc_roc': roc_auc_score(y_true, predictions['ml_anomaly_score'])
        }

        return metrics

    def save(self, path: str = "models/anomaly_detector.pkl"):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "models/anomaly_detector.pkl"):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {path}")
        return self

if __name__ == "__main__":
    import sys
    sys.path.append('.')

    # Load data
    merchants = pd.read_csv("data/raw/merchants.csv")
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=['timestamp'])

    # Initialize detector
    detector = AnomalyDetector()

    # Engineer features
    features = detector.engineer_features(merchants, transactions)

    # Train model
    detector.fit(features)

    # Predict
    predictions = detector.predict(features)

    # Evaluate
    metrics = detector.evaluate(features, merchants['is_fraud'])

    print("\nModel Performance:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Save predictions
    predictions.to_csv("data/processed/ml_predictions.csv", index=False)

    # Save model
    detector.save()
