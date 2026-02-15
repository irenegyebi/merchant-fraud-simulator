"""
Case Management System
Manages fraud investigation workflow and alert triage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid

class CaseManager:
    """Manages fraud investigation cases and review workflow"""

    def __init__(self):
        self.cases = {}
        self.case_counter = 0

    def create_case(self, merchant_id: str, risk_score: float, 
                   triggers: List[str], auto_escalate: bool = False) -> Dict:
        """Create a new investigation case"""
        self.case_counter += 1
        case_id = f"CASE_{self.case_counter:08d}"

        case = {
            'case_id': case_id,
            'merchant_id': merchant_id,
            'risk_score': risk_score,
            'triggers': triggers,
            'status': 'OPEN',
            'priority': 'HIGH' if auto_escalate else self._calculate_priority(risk_score),
            'created_at': datetime.now(),
            'assigned_to': None,
            'resolved_at': None,
            'resolution': None,
            'notes': [],
            'review_time_seconds': 0
        }

        self.cases[case_id] = case
        return case

    def _calculate_priority(self, risk_score: float) -> str:
        """Calculate case priority based on risk score"""
        if risk_score >= 90:
            return 'CRITICAL'
        elif risk_score >= 75:
            return 'HIGH'
        elif risk_score >= 60:
            return 'MEDIUM'
        else:
            return 'LOW'

    def assign_case(self, case_id: str, analyst_id: str) -> Dict:
        """Assign case to analyst"""
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")

        self.cases[case_id]['assigned_to'] = analyst_id
        self.cases[case_id]['notes'].append({
            'timestamp': datetime.now(),
            'analyst': 'SYSTEM',
            'note': f"Assigned to {analyst_id}"
        })

        return self.cases[case_id]

    def resolve_case(self, case_id: str, resolution: str, 
                    analyst_id: str, notes: str = "") -> Dict:
        """Resolve a case"""
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")

        case = self.cases[case_id]
        case['status'] = 'RESOLVED'
        case['resolution'] = resolution
        case['resolved_at'] = datetime.now()
        case['review_time_seconds'] = (
            case['resolved_at'] - case['created_at']
        ).total_seconds()

        case['notes'].append({
            'timestamp': datetime.now(),
            'analyst': analyst_id,
            'note': f"Resolved as {resolution}. {notes}"
        })

        return case

    def get_queue(self, status: str = 'OPEN', priority: str = None) -> pd.DataFrame:
        """Get cases in queue"""
        filtered_cases = [
            case for case in self.cases.values()
            if case['status'] == status
            and (priority is None or case['priority'] == priority)
        ]

        if not filtered_cases:
            return pd.DataFrame()

        return pd.DataFrame(filtered_cases)

    def get_metrics(self) -> Dict:
        """Get case management metrics"""
        if not self.cases:
            return {}

        cases_df = pd.DataFrame(self.cases.values())

        metrics = {
            'total_cases': len(cases_df),
            'open_cases': len(cases_df[cases_df['status'] == 'OPEN']),
            'resolved_cases': len(cases_df[cases_df['status'] == 'RESOLVED']),
            'avg_resolution_time': cases_df[
                cases_df['status'] == 'RESOLVED'
            ]['review_time_seconds'].mean(),
            'cases_by_priority': cases_df['priority'].value_counts().to_dict(),
            'cases_by_status': cases_df['status'].value_counts().to_dict(),
            'resolution_distribution': cases_df[
                cases_df['status'] == 'RESOLVED'
            ]['resolution'].value_counts().to_dict() if 'RESOLVED' in cases_df['status'].values else {}
        }

        return metrics

    def auto_create_cases(self, risk_scores_df: pd.DataFrame, 
                         threshold: float = 60) -> List[Dict]:
        """Automatically create cases for high-risk merchants"""
        high_risk = risk_scores_df[risk_scores_df['risk_score'] >= threshold]

        created_cases = []
        for _, merchant in high_risk.iterrows():
            # Parse risk factors
            triggers = merchant.get('risk_factors', '').split('; ') if pd.notna(merchant.get('risk_factors')) else ['High risk score']

            auto_escalate = merchant['risk_score'] >= 90

            case = self.create_case(
                merchant_id=merchant['merchant_id'],
                risk_score=merchant['risk_score'],
                triggers=triggers,
                auto_escalate=auto_escalate
            )
            created_cases.append(case)

        print(f"Created {len(created_cases)} cases from {len(high_risk)} high-risk merchants")
        return created_cases

class AlertMonitor:
    """Monitors system for fraud rate spikes and model drift"""

    def __init__(self, config_path: str = "config/config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.thresholds = config['monitoring']['alert_thresholds']
        self.alerts = []

    def check_fraud_rate(self, transactions_df: pd.DataFrame, 
                        window: str = '1h') -> List[Dict]:
        """Check for fraud rate spikes"""
        alerts = []

        # Resample by time window
        transactions_df = transactions_df.copy()
        transactions_df['window'] = transactions_df['timestamp'].dt.floor(window)

        fraud_rates = transactions_df.groupby('window').agg({
            'is_fraud': ['sum', 'count']
        }).reset_index()
        fraud_rates.columns = ['window', 'fraud_count', 'total_count']
        fraud_rates['fraud_rate'] = fraud_rates['fraud_count'] / fraud_rates['total_count']

        # Check for spikes
        spikes = fraud_rates[fraud_rates['fraud_rate'] > self.thresholds['fraud_rate_spike']]

        for _, spike in spikes.iterrows():
            alert = {
                'alert_id': f"ALERT_{len(self.alerts):06d}",
                'type': 'FRAUD_RATE_SPIKE',
                'severity': 'HIGH',
                'timestamp': spike['window'],
                'message': f"Fraud rate spike detected: {spike['fraud_rate']:.2%}",
                'details': {
                    'fraud_rate': spike['fraud_rate'],
                    'fraud_count': int(spike['fraud_count']),
                    'total_count': int(spike['total_count'])
                }
            }
            alerts.append(alert)
            self.alerts.append(alert)

        return alerts

    def check_decline_rate(self, decisions_df: pd.DataFrame) -> List[Dict]:
        """Check for decline rate spikes"""
        alerts = []

        decline_rate = (decisions_df['decision'] == 'DECLINE').mean()

        if decline_rate > self.thresholds['decline_rate_spike']:
            alert = {
                'alert_id': f"ALERT_{len(self.alerts):06d}",
                'type': 'DECLINE_RATE_SPIKE',
                'severity': 'MEDIUM',
                'timestamp': datetime.now(),
                'message': f"Decline rate spike: {decline_rate:.2%}",
                'details': {'decline_rate': decline_rate}
            }
            alerts.append(alert)
            self.alerts.append(alert)

        return alerts

    def calculate_psi(self, expected: pd.Series, actual: pd.Series, 
                     bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Calculate proportions
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                    np.log(actual_percents / expected_percents + 1e-10))

        return psi

    def check_model_drift(self, baseline_scores: pd.Series,
                         current_scores: pd.Series) -> Optional[Dict]:
        """Check for model drift using PSI"""
        psi = self.calculate_psi(baseline_scores, current_scores)

        if psi > self.thresholds['model_drift_psi']:
            alert = {
                'alert_id': f"ALERT_{len(self.alerts):06d}",
                'type': 'MODEL_DRIFT',
                'severity': 'HIGH' if psi > 0.35 else 'MEDIUM',
                'timestamp': datetime.now(),
                'message': f"Model drift detected (PSI: {psi:.3f})",
                'details': {'psi': psi, 'threshold': self.thresholds['model_drift_psi']}
            }
            self.alerts.append(alert)
            return alert

        return None

    def get_active_alerts(self) -> pd.DataFrame:
        """Get all active alerts"""
        return pd.DataFrame(self.alerts)

if __name__ == "__main__":
    # Test case manager
    cm = CaseManager()

    # Create sample cases
    cm.create_case("MERCH_00000001", 75.5, ["High chargeback rate", "New business"])
    cm.create_case("MERCH_00000002", 92.0, ["Velocity anomaly"], auto_escalate=True)

    print("Case Queue:")
    print(cm.get_queue())

    print("\nMetrics:")
    print(cm.get_metrics())
