"""
Merchant Profile Generator
Generates synthetic merchant data with realistic fraud patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Tuple
import uuid

class MerchantGenerator:
    """Generates synthetic merchant profiles with configurable fraud patterns"""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.merchant_config = self.config['data_generation']['merchant']
        self.random_seed = self.config['data_generation']['random_seed']
        np.random.seed(self.random_seed)

    def generate_merchants(self, n_merchants: int = None) -> pd.DataFrame:
        """Generate merchant profiles"""
        if n_merchants is None:
            n_merchants = self.config['data_generation']['n_merchants']

        print(f"Generating {n_merchants} merchant profiles...")

        merchants = []

        for i in range(n_merchants):
            merchant = self._generate_single_merchant(i)
            merchants.append(merchant)

        df = pd.DataFrame(merchants)

        # Inject fraud patterns
        df = self._inject_fraud_patterns(df)

        print(f"Generated {len(df)} merchants ({df['is_fraud'].sum()} fraudulent)")
        return df

    def _generate_single_merchant(self, idx: int) -> Dict:
        """Generate a single merchant profile"""
        # Select industry
        industry = self._select_industry()

        # Business age (days)
        business_age = int(np.random.exponential(365))
        business_age = max(self.merchant_config['business_age']['min_days'],
                          min(business_age, self.merchant_config['business_age']['max_days']))

        # Monthly revenue (log-normal)
        log_mean = np.log(50000)
        log_std = 1.5
        monthly_revenue = np.random.lognormal(log_mean, log_std)
        monthly_revenue = max(self.merchant_config['monthly_revenue']['min'],
                             min(monthly_revenue, self.merchant_config['monthly_revenue']['max']))

        # Chargeback rate
        base_chargeback = max(0, np.random.normal(
            self.merchant_config['chargeback_rate']['mean'],
            self.merchant_config['chargeback_rate']['std']
        ))

        # Geographic features
        country = self._select_country()
        region = self._select_region(country)

        # Business structure
        business_type = np.random.choice(
            ['Sole Proprietorship', 'LLC', 'Corporation', 'Partnership'],
            p=[0.3, 0.4, 0.2, 0.1]
        )

        # Website/digital presence
        has_website = np.random.choice([True, False], p=[0.8, 0.2])
        has_social_media = np.random.choice([True, False], p=[0.7, 0.3])

        return {
            'merchant_id': f"MERCH_{idx:08d}",
            'business_name': self._generate_business_name(industry),
            'industry': industry['name'],
            'industry_risk_weight': industry['risk_weight'],
            'business_type': business_type,
            'business_age_days': business_age,
            'monthly_revenue': round(monthly_revenue, 2),
            'chargeback_rate': round(base_chargeback, 4),
            'country': country,
            'region': region,
            'has_website': has_website,
            'has_social_media': has_social_media,
            'onboarding_date': datetime.now() - timedelta(days=business_age),
            'is_fraud': False,  # Will be set later
            'fraud_type': None,
            'risk_score': None
        }

    def _select_industry(self) -> Dict:
        """Select industry based on configured proportions"""
        industries = self.merchant_config['industries']
        probabilities = [ind['proportion'] for ind in industries]

        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]

        selected = np.random.choice(len(industries), p=probabilities)
        return industries[selected]

    def _select_country(self) -> str:
        """Select country with realistic distribution"""
        countries = ['US'] * 70 + ['CA'] * 10 + ['GB'] * 8 + ['DE'] * 5 + ['FR'] * 4 + ['Other'] * 3
        return np.random.choice(countries)

    def _select_region(self, country: str) -> str:
        """Select region based on country"""
        if country == 'US':
            regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
            return np.random.choice(regions)
        elif country == 'CA':
            return np.random.choice(['Ontario', 'Quebec', 'BC', 'Prairies'])
        else:
            return 'International'

    def _generate_business_name(self, industry: Dict) -> str:
        """Generate realistic business name"""
        prefixes = ['Global', 'Prime', 'Elite', 'Superior', 'Advanced', 'First', 'Pro', 'Smart']
        suffixes = ['Solutions', 'Services', 'Group', 'Corp', 'Inc', 'LLC', 'Ltd', 'Co']

        industry_words = {
            'Retail': ['Mart', 'Shop', 'Store', 'Goods', 'Market'],
            'E-commerce': ['Digital', 'Online', 'Web', 'Cyber', 'Virtual'],
            'Food & Beverage': ['Kitchen', 'Bistro', 'Eats', 'Cafe', 'Dining'],
            'Services': ['Consulting', 'Advisory', 'Support', 'Care'],
            'Travel': ['Journeys', 'Voyages', 'Trips', 'Destinations'],
            'Digital Goods': ['Software', 'Apps', 'Digital', 'Tech'],
            'Gambling': ['Bet', 'Wager', 'Gaming', 'Casino'],
            'Cryptocurrency': ['Crypto', 'Block', 'Chain', 'Coin']
        }

        words = industry_words.get(industry['name'], ['Business', 'Company'])

        name_type = np.random.choice(['prefix', 'suffix', 'combo'], p=[0.3, 0.3, 0.4])

        if name_type == 'prefix':
            return f"{np.random.choice(prefixes)} {np.random.choice(words)}"
        elif name_type == 'suffix':
            return f"{np.random.choice(words)} {np.random.choice(suffixes)}"
        else:
            return f"{np.random.choice(prefixes)} {np.random.choice(words)} {np.random.choice(suffixes)}"

    def _inject_fraud_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject realistic fraud patterns into merchant data"""
        n_fraud = int(len(df) * self.config['data_generation']['fraud_rate'])

        # Select fraud merchants
        fraud_indices = np.random.choice(df.index, size=n_fraud, replace=False)

        fraud_types = ['shell_company', 'transaction_laundering', 'bustout', 'identity_theft']

        for idx in fraud_indices:
            fraud_type = np.random.choice(fraud_types)
            df.at[idx, 'is_fraud'] = True
            df.at[idx, 'fraud_type'] = fraud_type

            # Modify features based on fraud type
            if fraud_type == 'shell_company':
                df.at[idx, 'business_age_days'] = np.random.randint(30, 90)
                df.at[idx, 'has_website'] = np.random.choice([True, False], p=[0.3, 0.7])
                df.at[idx, 'has_social_media'] = False
                df.at[idx, 'chargeback_rate'] *= 2

            elif fraud_type == 'transaction_laundering':
                df.at[idx, 'industry_risk_weight'] *= 1.5
                df.at[idx, 'monthly_revenue'] *= np.random.uniform(2, 5)
                df.at[idx, 'chargeback_rate'] *= 1.5

            elif fraud_type == 'bustout':
                df.at[idx, 'business_age_days'] = np.random.randint(180, 365)
                df.at[idx, 'chargeback_rate'] *= 3
                df.at[idx, 'monthly_revenue'] *= np.random.uniform(1.5, 3)

            elif fraud_type == 'identity_theft':
                df.at[idx, 'business_age_days'] = np.random.randint(30, 60)
                df.at[idx, 'has_website'] = False
                df.at[idx, 'chargeback_rate'] *= 4

        return df

if __name__ == "__main__":
    generator = MerchantGenerator()
    merchants = generator.generate_merchants()
    merchants.to_csv("data/raw/merchants.csv", index=False)
    print(f"Saved to data/raw/merchants.csv")
