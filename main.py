#!/usr/bin/env python3
"""
Merchant Fraud Detection Simulator
Main execution script

This script orchestrates the entire fraud detection pipeline:
1. Data generation
2. Risk scoring
3. ML model training
4. Case creation
5. Dashboard launch

Author: Irene A. Gyebi
Date: 2025
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

def setup_directories():
    """Create necessary directories"""
    dirs = ['data/raw', 'data/processed', 'models', 'logs', 'reports']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories created")

def generate_data():
    """Generate synthetic merchant and transaction data"""
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)

    from src.data_generation.merchant_generator import MerchantGenerator
    from src.data_generation.transaction_generator import TransactionGenerator

    # Generate merchants
    merchant_gen = MerchantGenerator()
    merchants = merchant_gen.generate_merchants()
    merchants.to_csv("data/raw/merchants.csv", index=False)
    print(f"Generated {len(merchants)} merchants")

    # Generate transactions
    txn_gen = TransactionGenerator()
    transactions = txn_gen.generate_transactions(merchants)
    transactions.to_csv("data/raw/transactions.csv", index=False)
    print(f"Generated {len(transactions)} transactions")

    return merchants, transactions

def run_risk_scoring(merchants, transactions):
    """Run rule-based risk scoring"""
    print("\n" + "="*60)
    print("STEP 2: RULE-BASED RISK SCORING")
    print("="*60)

    from src.risk_engine.rule_engine import RuleBasedRiskEngine

    engine = RuleBasedRiskEngine()
    risk_scores = engine.batch_score(merchants, transactions)
    risk_scores.to_csv("data/processed/merchant_risk_scores.csv", index=False)

    print("Risk scoring complete")
    print(f"   Approvals: {(risk_scores['decision'] == 'APPROVE').sum()}")
    print(f"   Reviews: {(risk_scores['decision'] == 'REVIEW').sum()}")
    print(f"   Declines: {(risk_scores['decision'] == 'DECLINE').sum()}")

    return risk_scores

def train_ml_model(merchants, transactions):
    """Train ML anomaly detection model"""
    print("\n" + "="*60)
    print("STEP 3: ML MODEL TRAINING")
    print("="*60)

    from src.models.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector()

    # Engineer features
    features = detector.engineer_features(merchants, transactions)

    # Train model
    detector.fit(features)

    # Predict
    predictions = detector.predict(features)
    predictions.to_csv("data/processed/ml_predictions.csv", index=False)

    # Evaluate
    metrics = detector.evaluate(features, merchants['is_fraud'])

    print("Model training complete")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1-Score: {metrics['f1_score']:.3f}")
    print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")

    # Save model
    detector.save()

    return predictions, metrics

def create_cases(risk_scores):
    """Create investigation cases for high-risk merchants"""
    print("\n" + "="*60)
    print("STEP 4: CASE MANAGEMENT")
    print("="*60)

    from src.monitoring.case_manager import CaseManager

    cm = CaseManager()
    cases = cm.auto_create_cases(risk_scores, threshold=60)

    print(f"Created {len(cases)} investigation cases")

    # Show metrics
    metrics = cm.get_metrics()
    print(f"   Critical Priority: {metrics.get('cases_by_priority', {}).get('CRITICAL', 0)}")
    print(f"   High Priority: {metrics.get('cases_by_priority', {}).get('HIGH', 0)}")

    return cases

def generate_report(merchants, transactions, risk_scores, ml_metrics):
    """Generate summary report"""
    print("\n" + "="*60)
    print("STEP 5: REPORT GENERATION")
    print("="*60)

    report = f"""
MERCHANT FRAUD DETECTION SIMULATOR - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================

DATA OVERVIEW
-------------
Total Merchants: {len(merchants):,}
Total Transactions: {len(transactions):,}
Fraud Rate: {merchants['is_fraud'].mean():.2%}

RISK SCORING RESULTS
--------------------
Approval Rate: {(risk_scores['decision'] == 'APPROVE').mean():.1%}
Review Rate: {(risk_scores['decision'] == 'REVIEW').mean():.1%}
Decline Rate: {(risk_scores['decision'] == 'DECLINE').mean():.1%}
Average Risk Score: {risk_scores['risk_score'].mean():.1f}

ML MODEL PERFORMANCE
--------------------
Precision: {ml_metrics['precision']:.3f}
Recall: {ml_metrics['recall']:.3f}
F1-Score: {ml_metrics['f1_score']:.3f}
AUC-ROC: {ml_metrics['auc_roc']:.3f}

TOP RISK INDUSTRIES
-------------------
"""

    fraud_by_industry = merchants.groupby('industry')['is_fraud'].mean().sort_values(ascending=False)
    for industry, rate in fraud_by_industry.head(5).items():
        report += f"{industry}: {rate:.2%}\n"

    report += """
RECOMMENDATIONS
---------------
1. Review all merchants in 'REVIEW' status within 24 hours
2. Investigate velocity anomalies in high-risk industries
3. Monitor model drift weekly and retrain monthly
4. Implement additional verification for shell company patterns

================================================================
"""

    with open("reports/executive_summary.txt", "w") as f:
        f.write(report)

    print("Report saved to reports/executive_summary.txt")
    print("\n" + report)

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("\n" + "="*60)
    print("STEP 6: LAUNCHING DASHBOARD")
    print("="*60)

    import subprocess
    print("Starting Streamlit dashboard...")
    print("Access at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard")
    print("="*60 + "\n")

    subprocess.run([
        "streamlit", "run", "dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Merchant Fraud Detection Simulator"
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip data generation (use existing data)"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training (use existing model)"
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Launch dashboard only"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("MERCHANT FRAUD DETECTION SIMULATOR")
    print("="*60)

    # Setup
    setup_directories()

    if args.dashboard_only:
        launch_dashboard()
        return

    # Step 1: Data Generation
    if args.skip_data and os.path.exists("data/raw/merchants.csv"):
        print("\nLoading existing data...")
        merchants = pd.read_csv("data/raw/merchants.csv")
        transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=['timestamp'])
        print(f"Loaded {len(merchants)} merchants, {len(transactions)} transactions")
    else:
        merchants, transactions = generate_data()

    # Step 2: Risk Scoring
    risk_scores = run_risk_scoring(merchants, transactions)

    merchants['merchant_id'] = merchants['merchant_id'].astype(str)
    risk_scores['merchant_id'] = risk_scores['merchant_id'].astype(str)
    # FIX: Merge the risk scores into the merchants dataframe
    # This ensures 'risk_score' is available for the dashboard and reports

    # Verify 'decision' and 'risk_score' exist before saving
    if 'decision' not in risk_scores.columns or 'risk_score' not in risk_scores.columns:
    # Force defaults if the engine failed to provide them
        risk_scores['decision'] = 'REVIEW'
        risk_scores['risk_score'] = 50.0
    
    # Save this explicitly so app.py can find it
    risk_scores.to_csv("data/processed/merchant_risk_scores.csv", index=False)

    merchants = merchants.merge(risk_scores[['merchant_id', 'risk_score', 'decision', 'risk_factors']],on='merchant_id', how='left')

    # Save the enriched version so the Dashboard can read it from the CSV
    merchants.to_csv("data/raw/merchants.csv", index=False)

    # Step 3: ML Model
    if args.skip_training and os.path.exists("models/anomaly_detector.pkl"):
        print("\nUsing existing model...")
        from src.models.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector()
        detector.load()
        features = detector.engineer_features(merchants, transactions)
        predictions = detector.predict(features)
        ml_metrics = detector.evaluate(features, merchants['is_fraud'])
    else:
        predictions, ml_metrics = train_ml_model(merchants, transactions)

    # Step 4: Case Management
    cases = create_cases(risk_scores)

    # Step 5: Report
    generate_report(merchants, transactions, risk_scores, ml_metrics)

    # Step 6: Dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
