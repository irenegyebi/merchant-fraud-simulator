"""
Transaction Generator
Generates synthetic transaction streams with velocity patterns and fraud signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from typing import Dict, List
import uuid

class TransactionGenerator:
    """Generates synthetic transaction data with realistic patterns"""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.random_seed = self.config['data_generation']['random_seed']
        np.random.seed(self.random_seed)

    def generate_transactions(self, merchants_df: pd.DataFrame, 
                            n_transactions: int = None) -> pd.DataFrame:
        """Generate transaction stream"""
        if n_transactions is None:
            n_transactions = self.config['data_generation']['n_transactions']

        print(f"Generating {n_transactions} transactions...")

        # Calculate transactions per merchant
        merchant_weights = self._calculate_merchant_weights(merchants_df)

        transactions = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)

        for i in range(n_transactions):
            # Select merchant
            merchant_idx = np.random.choice(len(merchants_df), p=merchant_weights)
            merchant = merchants_df.iloc[merchant_idx]

            # Generate transaction
            transaction = self._generate_single_transaction(
                merchant, start_time, end_time, i
            )
            transactions.append(transaction)

            if (i + 1) % 50000 == 0:
                print(f"  Generated {i + 1} transactions...")

        df = pd.DataFrame(transactions)

        # Add fraud labels based on merchant fraud status + transaction patterns
        df = self._label_fraudulent_transactions(df, merchants_df)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Generated {len(df)} transactions ({df['is_fraud'].sum()} fraudulent)")
        return df

    def _calculate_merchant_weights(self, merchants_df: pd.DataFrame) -> np.ndarray:
        """Calculate transaction frequency weights for each merchant"""
        # Higher revenue = more transactions
        weights = merchants_df['monthly_revenue'].values

        # Fraudulent merchants often have velocity spikes
        fraud_multiplier = merchants_df['is_fraud'].apply(
            lambda x: np.random.uniform(1.5, 3.0) if x else 1.0
        ).values

        weights = weights * fraud_multiplier
        return weights / weights.sum()

    def _generate_single_transaction(self, merchant: pd.Series, 
                                   start_time: datetime, end_time: datetime,
                                   idx: int) -> Dict:
        """Generate a single transaction"""
        # Timestamp
        time_range = (end_time - start_time).total_seconds()
        random_seconds = np.random.randint(0, int(time_range))
        timestamp = start_time + timedelta(seconds=random_seconds)

        # Amount (log-normal distribution)
        base_amount = np.random.lognormal(4, 1.5)  # median around $55

        # Adjust amount based on industry and fraud
        if merchant['is_fraud']:
            # Fraudulent transactions often have unusual amounts
            if np.random.random() < 0.3:
                # Card testing: very small amounts
                amount = np.random.uniform(0.5, 2.0)
            elif np.random.random() < 0.2:
                # Large purchases: unusually high
                amount = base_amount * np.random.uniform(3, 10)
            else:
                amount = base_amount
        else:
            amount = base_amount

        # Card present vs online
        card_present = np.random.choice([True, False], p=[0.3, 0.7])

        # Currency
        currency = 'USD' if merchant['country'] == 'US' else                   'CAD' if merchant['country'] == 'CA' else 'EUR'

        # Payment method
        payment_method = np.random.choice(
            ['credit_card', 'debit_card', 'digital_wallet'],
            p=[0.6, 0.25, 0.15]
        )

        # Device fingerprint (simplified)
        device_fingerprint = hash(f"{merchant['merchant_id']}_{idx}") % 1000000

        # Customer ID (some repeat customers)
        if np.random.random() < 0.3:
            # Returning customer
            customer_id = f"CUST_{np.random.randint(1, 50000):08d}"
        else:
            # New customer
            customer_id = f"CUST_{np.random.randint(50000, 100000):08d}"

        return {
            'transaction_id': f"TXN_{idx:010d}",
            'merchant_id': merchant['merchant_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'currency': currency,
            'payment_method': payment_method,
            'card_present': card_present,
            'customer_id': customer_id,
            'device_fingerprint': device_fingerprint,
            'is_fraud': False,  # Will be determined later
            'fraud_reason': None
        }

    def _label_fraudulent_transactions(self, transactions_df: pd.DataFrame,
                                     merchants_df: pd.DataFrame) -> pd.DataFrame:
        """Label transactions as fraudulent based on merchant status and patterns"""
        # Create merchant fraud lookup
        merchant_fraud = merchants_df.set_index('merchant_id')['is_fraud'].to_dict()
        merchant_fraud_type = merchants_df.set_index('merchant_id')['fraud_type'].to_dict()

        # Base fraud label from merchant
        transactions_df['is_fraud'] = transactions_df['merchant_id'].map(merchant_fraud)
        transactions_df['fraud_reason'] = transactions_df['merchant_id'].map(merchant_fraud_type)

        # Add velocity-based fraud patterns
        transactions_df = self._add_velocity_fraud(transactions_df)

        # Add geographic fraud patterns
        transactions_df = self._add_geographic_fraud(transactions_df)

        return transactions_df

    def _add_velocity_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity-based fraud patterns"""
        # Calculate velocity features
        df = df.sort_values(['merchant_id', 'timestamp'])
        df['hour'] = df['timestamp'].dt.floor('H')

        # Count transactions per merchant per hour
        velocity = df.groupby(['merchant_id', 'hour']).size().reset_index(name='hourly_count')

        # Flag velocity anomalies
        velocity_threshold = self.config['risk_engine']['velocity']['hourly_transaction_limit']

        high_velocity = velocity[velocity['hourly_count'] > velocity_threshold]

        for _, row in high_velocity.iterrows():
            mask = (df['merchant_id'] == row['merchant_id']) &                    (df['hour'] == row['hour'])

            # Flag some of these as fraud (not all, for realism)
            fraud_mask = mask & (np.random.random(mask.sum()) < 0.3)
            df.loc[fraud_mask, 'is_fraud'] = True
            df.loc[fraud_mask, 'fraud_reason'] = 'velocity_anomaly'

        df = df.drop('hour', axis=1)
        return df

    def _add_geographic_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geographic inconsistency fraud patterns"""
        # Simulate some transactions with impossible geography
        # (simplified: same customer, different countries within short time)

        df = df.sort_values(['customer_id', 'timestamp'])
        df['prev_merchant'] = df.groupby('customer_id')['merchant_id'].shift(1)
        df['time_diff'] = df.groupby('customer_id')['timestamp'].diff()

        # Find suspicious patterns (would need merchant country lookup in real implementation)
        # For simulation, randomly flag some as geographic fraud
        suspicious = (df['time_diff'] < timedelta(hours=1)) &                     (np.random.random(len(df)) < 0.05)

        df.loc[suspicious, 'is_fraud'] = True
        df.loc[suspicious, 'fraud_reason'] = 'geographic_inconsistency'

        df = df.drop(['prev_merchant', 'time_diff'], axis=1)
        return df

if __name__ == "__main__":
    import sys
    sys.path.append('.')

    # Load merchants
    merchants = pd.read_csv("data/raw/merchants.csv")

    generator = TransactionGenerator()
    transactions = generator.generate_transactions(merchants)
    transactions.to_csv("data/raw/transactions.csv", index=False)
    print(f"Saved to data/raw/transactions.csv")
