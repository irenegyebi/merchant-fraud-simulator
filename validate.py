#!/usr/bin/env python3
"""
Project Validation Script
Checks that all required files exist and are valid
"""

import os
import sys
import yaml

def check_file_exists(filepath, required=True):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = "(required)" if required else "(optional)"
    print(f"{status} {filepath} {req_text}")
    return exists

def check_python_syntax(filepath):
    """Check Python file for syntax errors"""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False

def check_yaml_valid(filepath):
    """Check YAML file is valid"""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        return True
    except yaml.YAMLError as e:
        print(f"   ❌ YAML error: {e}")
        return False

def main():
    print("="*60)
    print("Merchant Fraud Detection Simulator - Project Validation")
    print("="*60)
    print()

    required_files = [
        "config/config.yaml",
        "src/data_generation/merchant_generator.py",
        "src/data_generation/transaction_generator.py",
        "src/risk_engine/rule_engine.py",
        "src/models/anomaly_detector.py",
        "src/monitoring/case_manager.py",
        "dashboard/app.py",
        "main.py",
        "requirements.txt",
        "README.md",
        "Dockerfile",
    ]

    optional_files = [
        "docker-compose.yml",
        "quick_start.sh",
    ]

    print("Checking required files...")
    all_required_present = True
    for filepath in required_files:
        if not check_file_exists(filepath, required=True):
            all_required_present = False

    print()
    print("Checking optional files...")
    for filepath in optional_files:
        check_file_exists(filepath, required=False)

    print()
    print("Validating file contents...")

    # Check Python syntax
    python_files = [
        "src/data_generation/merchant_generator.py",
        "src/data_generation/transaction_generator.py",
        "src/risk_engine/rule_engine.py",
        "src/models/anomaly_detector.py",
        "src/monitoring/case_manager.py",
        "dashboard/app.py",
        "main.py",
    ]

    all_python_valid = True
    for filepath in python_files:
        if os.path.exists(filepath):
            if not check_python_syntax(filepath):
                all_python_valid = False
            else:
                print(f"   ✅ {filepath} - valid Python")

    # Check YAML validity
    if os.path.exists("config/config.yaml"):
        if check_yaml_valid("config/config.yaml"):
            print(f"   ✅ config/config.yaml - valid YAML")
        else:
            print(f"   ❌ config/config.yaml - invalid YAML")

    print()
    print("="*60)

    if all_required_present and all_python_valid:
        print("✅ PROJECT VALIDATION PASSED")
        print()
        print("Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run tests: python tests/test_fraud_detection.py")
        print("  3. Launch system: python main.py")
        print()
        return 0
    else:
        print("❌ PROJECT VALIDATION FAILED")
        print()
        print("Please fix the issues above before proceeding.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
