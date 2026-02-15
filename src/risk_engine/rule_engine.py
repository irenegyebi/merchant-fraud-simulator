"""
Rule-Based Risk Scoring Engine
Implements merchant risk scoring using configurable rules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Tuple

class RuleBasedRiskEngine:
    """Rule-based risk scoring for merchant underwriting and monitoring"""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.rule_weights = self.config['risk_engine']['rule_weights']
        self.thresholds = self.config['risk_engine']['thresholds']
        self.velocity_limits = self.config['risk_engine']['velocity']

    def calculate_risk_score(self, merchant: pd.Series, 
                           transaction_history: pd.DataFrame = None) -> Dict:
        """Calculate comprehensive risk score for a merchant"""

        scores = {}

        # Industry risk score (0-100)
        scores['industry_risk'] = self._calculate_industry_risk(merchant)

        # Business age score (0-100, inverse: newer = higher risk)
        scores['business_age'] = self._calculate_business_age_risk(merchant)

        # Chargeback history score (0-100)
        scores['chargeback_history'] = self._calculate_chargeback_risk(merchant)

        # Velocity score (0-100, requires transaction history)
        if transaction_history is not None:
            scores['velocity'] = self._calculate_velocity_risk(merchant, transaction_history)
        else:
            scores['velocity'] = 50  # Neutral if no data

        # Geographic consistency score
        scores['geographic_consistency'] = self._calculate_geographic_risk(merchant)

        # Transaction pattern score
        if transaction_history is not None:
            scores['transaction_pattern'] = self._calculate_pattern_risk(merchant, transaction_history)
        else:
            scores['transaction_pattern'] = 50

        # Calculate weighted composite score
        composite_score = sum(
            scores[rule] * self.rule_weights[rule] 
            for rule in scores.keys()
        )

        # Determine decision
        decision = self._get_decision(composite_score)

        return {
            'merchant_id': merchant['merchant_id'],
            'composite_score': round(composite_score, 2),
            'decision': decision,
            'rule_scores': scores,
            'risk_factors': self._identify_risk_factors(scores, merchant),
            'timestamp': datetime.now()
        }

    def _calculate_industry_risk(self, merchant: pd.Series) -> float:
        """Calculate industry-based risk score"""
        risk_weight = merchant['industry_risk_weight']
        # Scale to 0-100 (higher weight = higher risk)
        base_score = (risk_weight - 0.9) / (3.0 - 0.9) * 100
        return min(100, max(0, base_score))

    def _calculate_business_age_risk(self, merchant: pd.Series) -> float:
        """Calculate business age risk (newer businesses are riskier)"""
        age_days = merchant['business_age_days']

        # Score decreases as age increases
        if age_days < 90:
            return 80  # Very high risk
        elif age_days < 180:
            return 60
        elif age_days < 365:
            return 40
        elif age_days < 730:
            return 25
        else:
            return 15  # Established business

    def _calculate_chargeback_risk(self, merchant: pd.Series) -> float:
        """Calculate chargeback rate risk"""
        chargeback_rate = merchant['chargeback_rate']

        # Industry average ~1%, >3% is concerning
        if chargeback_rate < 0.005:
            return 10
        elif chargeback_rate < 0.01:
            return 30
        elif chargeback_rate < 0.02:
            return 50
        elif chargeback_rate < 0.05:
            return 75
        else:
            return 95

    def _calculate_velocity_risk(self, merchant: pd.Series, 
                                transactions: pd.DataFrame) -> float:
        """Calculate transaction velocity risk"""
        merchant_txns = transactions[transactions['merchant_id'] == merchant['merchant_id']]

        if len(merchant_txns) == 0:
            return 50

        # Calculate velocity metrics
        recent_txns = merchant_txns[
            merchant_txns['timestamp'] > datetime.now() - timedelta(hours=24)
        ]

        hourly_counts = recent_txns.groupby(
            recent_txns['timestamp'].dt.floor('H')
        ).size()

        max_hourly = hourly_counts.max() if len(hourly_counts) > 0 else 0

        # Score based on velocity limits
        if max_hourly > self.velocity_limits['hourly_transaction_limit'] * 2:
            return 90
        elif max_hourly > self.velocity_limits['hourly_transaction_limit']:
            return 70
        elif max_hourly > self.velocity_limits['hourly_transaction_limit'] * 0.5:
            return 50
        else:
            return 20

    def _calculate_geographic_risk(self, merchant: pd.Series) -> float:
        """Calculate geographic risk score"""
        country = merchant['country']

        # Simplified risk by country
        country_risk = {
            'US': 20,
            'CA': 25,
            'GB': 30,
            'DE': 30,
            'FR': 35,
            'Other': 60
        }

        base_score = country_risk.get(country, 50)

        # Adjust for digital presence
        if not merchant['has_website']:
            base_score += 15
        if not merchant['has_social_media']:
            base_score += 10

        return min(100, base_score)

    def _calculate_pattern_risk(self, merchant: pd.Series,
                              transactions: pd.DataFrame) -> float:
        """Calculate transaction pattern risk"""
        merchant_txns = transactions[transactions['merchant_id'] == merchant['merchant_id']]

        if len(merchant_txns) < 10:
            return 50  # Insufficient data

        risk_score = 0

        # Check for amount clustering (round numbers are suspicious)
        amounts = merchant_txns['amount']
        round_amount_pct = (amounts == amounts.round()).mean()
        if round_amount_pct > 0.3:
            risk_score += 20

        # Check for card testing pattern (small amounts)
        small_txns = (amounts < 5).mean()
        if small_txns > 0.1:
            risk_score += 25

        # Check for unusual hours
        merchant_txns['hour'] = merchant_txns['timestamp'].dt.hour
        night_txns = ((merchant_txns['hour'] < 6) | (merchant_txns['hour'] > 23)).mean()
        if night_txns > 0.3:
            risk_score += 15

        return min(100, risk_score)

    def _get_decision(self, score: float) -> str:
        """Get decision based on score thresholds"""
        if score < self.thresholds['approve']:
            return 'APPROVE'
        elif score < self.thresholds['review']:
            return 'REVIEW'
        else:
            return 'DECLINE'

    def _identify_risk_factors(self, scores: Dict, merchant: pd.Series) -> List[str]:
        """Identify specific risk factors for explanation"""
        factors = []

        if scores['industry_risk'] > 60:
            factors.append(f"High-risk industry: {merchant['industry']}")

        if scores['business_age'] > 60:
            factors.append(f"New business: {merchant['business_age_days']} days")

        if scores['chargeback_history'] > 60:
            factors.append(f"Elevated chargeback rate: {merchant['chargeback_rate']:.2%}")

        if scores['velocity'] > 60:
            factors.append("Unusual transaction velocity")

        if not merchant['has_website']:
            factors.append("No website presence")

        return factors

    def batch_score(self, merchants_df: pd.DataFrame, 
                   transactions_df: pd.DataFrame = None) -> pd.DataFrame:
        """Score multiple merchants"""
        results = []

        for idx, merchant in merchants_df.iterrows():
            # Get transaction history for this merchant
            merchant_txns = None
            if transactions_df is not None:
                merchant_txns = transactions_df[
                    transactions_df['merchant_id'] == merchant['merchant_id']
                ]

            result = self.calculate_risk_score(merchant, merchant_txns)
            results.append(result)

            if (idx + 1) % 1000 == 0:
                print(f"Scored {idx + 1} merchants...")

        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                'merchant_id': r['merchant_id'],
                'risk_score': r['composite_score'],
                'decision': r['decision'],
                'industry_risk': r['rule_scores']['industry_risk'],
                'business_age_risk': r['rule_scores']['business_age'],
                'chargeback_risk': r['rule_scores']['chargeback_history'],
                'velocity_risk': r['rule_scores']['velocity'],
                'geographic_risk': r['rule_scores']['geographic_consistency'],
                'pattern_risk': r['rule_scores']['transaction_pattern'],
                'risk_factors': '; '.join(r['risk_factors']),
                'scored_at': r['timestamp']
            }
            for r in results
        ])

        return results_df

if __name__ == "__main__":
    import sys
    sys.path.append('.')

    # Load data
    merchants = pd.read_csv("data/raw/merchants.csv")
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=['timestamp'])

    # Initialize engine
    engine = RuleBasedRiskEngine()

    # Score merchants
    scores = engine.batch_score(merchants, transactions)
    scores.to_csv("data/processed/merchant_risk_scores.csv", index=False)

    print("\nRisk Score Distribution:")
    print(scores['decision'].value_counts())
    print(f"\nAverage Risk Score: {scores['risk_score'].mean():.2f}")
