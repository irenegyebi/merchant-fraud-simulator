"""
Test Suite for Merchant Fraud Detection Simulator
Comprehensive tests for all core components
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation.merchant_generator import MerchantGenerator
from src.data_generation.transaction_generator import TransactionGenerator
from src.risk_engine.rule_engine import RuleBasedRiskEngine
from src.models.anomaly_detector import AnomalyDetector
from src.monitoring.case_manager import CaseManager, AlertMonitor

class TestDataGeneration(unittest.TestCase):
    """Test data generation components"""

    def setUp(self):
        self.merchant_gen = MerchantGenerator()
        self.transaction_gen = TransactionGenerator()

    def test_merchant_generation(self):
        """Test merchant profile generation"""
        merchants = self.merchant_gen.generate_merchants(n_merchants=100)

        self.assertEqual(len(merchants), 100)
        self.assertIn('merchant_id', merchants.columns)
        self.assertIn('is_fraud', merchants.columns)
        self.assertIn('fraud_type', merchants.columns)

        # Check fraud rate is approximately correct
        fraud_rate = merchants['is_fraud'].mean()
        self.assertGreater(fraud_rate, 0.03)
        self.assertLess(fraud_rate, 0.07)

    def test_merchant_features(self):
        """Test merchant feature ranges"""
        merchants = self.merchant_gen.generate_merchants(n_merchants=50)

        # Business age should be positive
        self.assertTrue(all(merchants['business_age_days'] > 0))

        # Revenue should be positive
        self.assertTrue(all(merchants['monthly_revenue'] > 0))

        # Chargeback rate should be non-negative
        self.assertTrue(all(merchants['chargeback_rate'] >= 0))

        # Industry risk weights should be reasonable
        self.assertTrue(all(merchants['industry_risk_weight'] >= 0.9))
        self.assertTrue(all(merchants['industry_risk_weight'] <= 3.0))

    def test_transaction_generation(self):
        """Test transaction generation"""
        merchants = self.merchant_gen.generate_merchants(n_merchants=10)
        transactions = self.transaction_gen.generate_transactions(
            merchants, n_transactions=1000
        )

        self.assertEqual(len(transactions), 1000)
        self.assertIn('transaction_id', transactions.columns)
        self.assertIn('is_fraud', transactions.columns)

        # All transactions should link to valid merchants
        self.assertTrue(
            all(transactions['merchant_id'].isin(merchants['merchant_id']))
        )

    def test_transaction_amounts(self):
        """Test transaction amount distributions"""
        merchants = self.merchant_gen.generate_merchants(n_merchants=10)
        transactions = self.transaction_gen.generate_transactions(
            merchants, n_transactions=500
        )

        # Amounts should be positive
        self.assertTrue(all(transactions['amount'] > 0))

        # Should have some variation
        self.assertGreater(transactions['amount'].std(), 0)

class TestRiskEngine(unittest.TestCase):
    """Test rule-based risk scoring engine"""

    def setUp(self):
        self.engine = RuleBasedRiskEngine()

        # Create test merchant
        self.test_merchant = pd.Series({
            'merchant_id': 'TEST_001',
            'industry': 'Retail',
            'industry_risk_weight': 1.0,
            'business_age_days': 365,
            'monthly_revenue': 50000,
            'chargeback_rate': 0.01,
            'country': 'US',
            'region': 'Northeast',
            'has_website': True,
            'has_social_media': True
        })

    def test_industry_risk_calculation(self):
        """Test industry risk scoring"""
        low_risk = self.engine._calculate_industry_risk(
            pd.Series({'industry_risk_weight': 0.9})
        )
        high_risk = self.engine._calculate_industry_risk(
            pd.Series({'industry_risk_weight': 3.0})
        )

        self.assertLess(low_risk, high_risk)
        self.assertGreaterEqual(low_risk, 0)
        self.assertLessEqual(high_risk, 100)

    def test_business_age_risk(self):
        """Test business age risk scoring"""
        new_business = self.engine._calculate_business_age_risk(
            pd.Series({'business_age_days': 30})
        )
        old_business = self.engine._calculate_business_age_risk(
            pd.Series({'business_age_days': 1000})
        )

        self.assertGreater(new_business, old_business)
        self.assertEqual(new_business, 80)  # Very high risk
        self.assertEqual(old_business, 15)  # Established

    def test_chargeback_risk(self):
        """Test chargeback risk scoring"""
        low_cb = self.engine._calculate_chargeback_risk(
            pd.Series({'chargeback_rate': 0.001})
        )
        high_cb = self.engine._calculate_chargeback_risk(
            pd.Series({'chargeback_rate': 0.10})
        )

        self.assertLess(low_cb, high_cb)
        self.assertEqual(low_cb, 10)
        self.assertEqual(high_cb, 95)

    def test_decision_thresholds(self):
        """Test decision threshold logic"""
        self.assertEqual(self.engine._get_decision(20), 'APPROVE')
        self.assertEqual(self.engine._get_decision(45), 'REVIEW')
        self.assertEqual(self.engine._get_decision(75), 'DECLINE')

    def test_full_scoring(self):
        """Test complete risk scoring"""
        result = self.engine.calculate_risk_score(self.test_merchant)

        self.assertIn('composite_score', result)
        self.assertIn('decision', result)
        self.assertIn('rule_scores', result)
        self.assertIn('risk_factors', result)

        self.assertGreaterEqual(result['composite_score'], 0)
        self.assertLessEqual(result['composite_score'], 100)
        self.assertIn(result['decision'], ['APPROVE', 'REVIEW', 'DECLINE'])

class TestAnomalyDetector(unittest.TestCase):
    """Test ML anomaly detection model"""

    def setUp(self):
        self.detector = AnomalyDetector()

        # Generate small test dataset
        self.merchants = pd.DataFrame({
            'merchant_id': [f'MERCH_{i:03d}' for i in range(100)],
            'industry_risk_weight': np.random.uniform(0.9, 2.0, 100),
            'business_age_days': np.random.randint(30, 1000, 100),
            'monthly_revenue': np.random.lognormal(10, 1, 100),
            'chargeback_rate': np.random.exponential(0.01, 100),
            'has_website': np.random.choice([True, False], 100),
            'has_social_media': np.random.choice([True, False], 100),
            'is_fraud': np.random.choice([True, False], 100, p=[0.05, 0.95])
        })

        # Create dummy transaction features
        self.features = self.merchants.copy()
        self.features['txn_count'] = np.random.randint(10, 1000, 100)
        self.features['avg_transaction_amount'] = np.random.lognormal(4, 1, 100)
        self.features['std_transaction_amount'] = np.random.exponential(50, 100)
        self.features['max_transaction_amount'] = self.features['avg_transaction_amount'] * 3
        self.features['velocity_1h_max'] = np.random.randint(1, 50, 100)
        self.features['velocity_24h_count'] = np.random.randint(10, 500, 100)
        self.features['unique_customers'] = np.random.randint(5, 200, 100)
        self.features['repeat_customer_rate'] = np.random.uniform(0, 1, 100)
        self.features['card_present_ratio'] = np.random.uniform(0, 1, 100)
        self.features['night_transaction_ratio'] = np.random.uniform(0, 0.5, 100)

    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        # This would require actual transaction data
        # For now, just test that detector initializes correctly
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.scaler)

    def test_model_training(self):
        """Test model training"""
        self.detector.feature_columns = [
            'industry_risk_weight', 'business_age_days', 'monthly_revenue',
            'chargeback_rate', 'has_website', 'has_social_media',
            'txn_count', 'avg_transaction_amount', 'std_transaction_amount',
            'max_transaction_amount', 'velocity_1h_max', 'velocity_24h_count',
            'unique_customers', 'repeat_customer_rate', 'card_present_ratio',
            'night_transaction_ratio'
        ]

        # Should not raise exception
        try:
            self.detector.fit(self.features)
        except Exception as e:
            self.fail(f"Model training failed: {e}")

    def test_prediction_output(self):
        """Test prediction output format"""
        self.detector.feature_columns = [
            'industry_risk_weight', 'business_age_days', 'monthly_revenue',
            'chargeback_rate', 'has_website', 'has_social_media',
            'txn_count', 'avg_transaction_amount', 'std_transaction_amount',
            'max_transaction_amount', 'velocity_1h_max', 'velocity_24h_count',
            'unique_customers', 'repeat_customer_rate', 'card_present_ratio',
            'night_transaction_ratio'
        ]

        self.detector.fit(self.features)
        predictions = self.detector.predict(self.features)

        self.assertEqual(len(predictions), len(self.features))
        self.assertIn('ml_anomaly_score', predictions.columns)
        self.assertIn('is_anomaly', predictions.columns)

        # Scores should be between 0 and 100
        self.assertTrue(all(predictions['ml_anomaly_score'] >= 0))
        self.assertTrue(all(predictions['ml_anomaly_score'] <= 100))

class TestCaseManager(unittest.TestCase):
    """Test case management system"""

    def setUp(self):
        self.cm = CaseManager()

    def test_case_creation(self):
        """Test case creation"""
        case = self.cm.create_case(
            merchant_id="TEST_001",
            risk_score=75.0,
            triggers=["High chargeback rate", "Velocity anomaly"],
            auto_escalate=False
        )

        self.assertIn('case_id', case)
        self.assertEqual(case['merchant_id'], "TEST_001")
        self.assertEqual(case['status'], 'OPEN')
        self.assertEqual(case['priority'], 'HIGH')
        self.assertEqual(len(case['triggers']), 2)

    def test_auto_escalation(self):
        """Test automatic escalation for high scores"""
        case = self.cm.create_case(
            merchant_id="TEST_002",
            risk_score=92.0,
            triggers=["Critical risk"],
            auto_escalate=True
        )

        self.assertEqual(case['priority'], 'CRITICAL')

    def test_case_resolution(self):
        """Test case resolution workflow"""
        case = self.cm.create_case(
            merchant_id="TEST_003",
            risk_score=65.0,
            triggers=["Medium risk"]
        )

        case_id = case['case_id']

        # Assign
        self.cm.assign_case(case_id, "analyst_001")
        self.assertEqual(self.cm.cases[case_id]['assigned_to'], "analyst_001")

        # Resolve
        self.cm.resolve_case(case_id, "LEGITIMATE", "analyst_001", "Verified business")
        self.assertEqual(self.cm.cases[case_id]['status'], 'RESOLVED')
        self.assertEqual(self.cm.cases[case_id]['resolution'], 'LEGITIMATE')

    def test_queue_management(self):
        """Test queue filtering"""
        # Create cases with different priorities
        self.cm.create_case("M1", 95.0, ["Critical"], auto_escalate=True)
        self.cm.create_case("M2", 80.0, ["High"])
        self.cm.create_case("M3", 65.0, ["Medium"])

        # Get high priority queue
        high_priority = self.cm.get_queue(priority='HIGH')
        self.assertGreater(len(high_priority), 0)

        # Get all open cases
        all_open = self.cm.get_queue()
        self.assertEqual(len(all_open), 3)

    def test_metrics(self):
        """Test metrics calculation"""
        # Create and resolve some cases
        case1 = self.cm.create_case("M1", 70.0, ["Test"])
        self.cm.resolve_case(case1['case_id'], "FRAUD", "analyst_001")

        case2 = self.cm.create_case("M2", 75.0, ["Test"])

        metrics = self.cm.get_metrics()

        self.assertEqual(metrics['total_cases'], 2)
        self.assertEqual(metrics['resolved_cases'], 1)
        self.assertEqual(metrics['open_cases'], 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution"""
        # Generate small dataset
        merchant_gen = MerchantGenerator()
        merchants = merchant_gen.generate_merchants(n_merchants=50)

        transaction_gen = TransactionGenerator()
        transactions = transaction_gen.generate_transactions(
            merchants, n_transactions=500
        )

        # Risk scoring
        engine = RuleBasedRiskEngine()
        risk_scores = engine.batch_score(merchants, transactions)

        # Verify outputs
        self.assertEqual(len(risk_scores), len(merchants))
        self.assertTrue(all(risk_scores['risk_score'] >= 0))
        self.assertTrue(all(risk_scores['risk_score'] <= 100))

        # Case creation
        cm = CaseManager()
        cases = cm.auto_create_cases(risk_scores, threshold=60)

        # Should create cases for high-risk merchants
        high_risk_count = len(risk_scores[risk_scores['risk_score'] >= 60])
        self.assertEqual(len(cases), high_risk_count)

if __name__ == '__main__':
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")

    sys.exit(0 if result.wasSuccessful() else 1)
