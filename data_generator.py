import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

class DomainSchemaInferrer:
    """Infers realistic column names and data types based on project details"""
    
    # Predefined domain schemas: (Column Name, Type, Distribution/Logic)
    DOMAIN_SCHEMAS = {
        'ecommerce': [
            ('product_id', 'categorical', ['P1001', 'P1002', 'P1003', 'P1004', 'P1005']),
            ('category', 'categorical', ['Electronics', 'Home & Kitchen', 'Beauty', 'Books', 'Clothing']),
            ('price', 'numerical', {'mean': 150, 'std': 100, 'min': 5}),
            ('quantity', 'numerical', {'mean': 2, 'std': 1, 'min': 1, 'max': 10}),
            ('customer_rating', 'numerical', {'mean': 4.2, 'std': 0.8, 'min': 1, 'max': 5}),
            ('in_stock', 'categorical', ['Yes', 'No']),
            ('discount_applied', 'categorical', ['0%', '10%', '20%', '30%', '50%'])
        ],
        'healthcare': [
            ('patient_id', 'categorical', [f'PT{i:04d}' for i in range(100)]),
            ('age', 'numerical', {'mean': 45, 'std': 20, 'min': 1, 'max': 95}),
            ('gender', 'categorical', ['Male', 'Female', 'Other']),
            ('blood_pressure_systolic', 'numerical', {'mean': 120, 'std': 15, 'min': 90, 'max': 180}),
            ('cholesterol', 'numerical', {'mean': 200, 'std': 40, 'min': 120, 'max': 300}),
            ('diagnosis', 'categorical', ['Healthy', 'Hypertension', 'Diabetes', 'Cardiac Issue', 'Infection']),
            ('treatment_type', 'categorical', ['Medication', 'Surgery', 'Physical Therapy', 'Observation'])
        ],
        'finance': [
            ('transaction_id', 'categorical', [f'TXN{i:06d}' for i in range(100)]),
            ('account_type', 'categorical', ['Savings', 'Current', 'Credit Card', 'Loan', 'Fixed Deposit']),
            ('amount', 'numerical', {'mean': 5000, 'std': 10000, 'min': 10}),
            ('credit_score', 'numerical', {'mean': 700, 'std': 50, 'min': 300, 'max': 850}),
            ('is_fraud', 'categorical', ['No', 'Yes']),
            ('currency', 'categorical', ['USD', 'EUR', 'GBP', 'INR', 'JPY'])
        ],
        'education': [
            ('student_id', 'categorical', [f'STU{i:05d}' for i in range(100)]),
            ('grade_level', 'categorical', ['9th', '10th', '11th', '12th']),
            ('attendance_rate', 'numerical', {'mean': 92, 'std': 5, 'min': 50, 'max': 100}),
            ('midterm_score', 'numerical', {'mean': 75, 'std': 15, 'min': 0, 'max': 100}),
            ('final_score', 'numerical', {'mean': 78, 'std': 18, 'min': 0, 'max': 100}),
            ('subject', 'categorical', ['Math', 'Science', 'History', 'English', 'Computer Science']),
            ('passed', 'categorical', ['Yes', 'No'])
        ],
        'social_media': [
            ('user_id', 'categorical', [f'USR{i:06d}' for i in range(100)]),
            ('post_type', 'categorical', ['Image', 'Video', 'Text', 'Story', 'Reel']),
            ('likes', 'numerical', {'mean': 500, 'std': 2000, 'min': 0}),
            ('shares', 'numerical', {'mean': 50, 'std': 300, 'min': 0}),
            ('comments', 'numerical', {'mean': 30, 'std': 100, 'min': 0}),
            ('engagement_score', 'numerical', {'mean': 0.05, 'std': 0.1, 'min': 0, 'max': 1}),
            ('platform', 'categorical', ['Instagram', 'TikTok', 'X', 'Facebook', 'LinkedIn'])
        ]
    }

    @staticmethod
    def infer_domain(project_details):
        if not project_details:
            return 'generic'
        
        details = project_details.lower()
        
        # Keyword mapping
        mappings = {
            'ecommerce': ['shop', 'store', 'product', 'sales', 'customer', 'order', 'retail', 'price'],
            'healthcare': ['health', 'patient', 'hospital', 'doctor', 'medical', 'diagnosis', 'disease', 'clinic'],
            'finance': ['bank', 'loan', 'credit', 'transaction', 'stock', 'finance', 'revenue', 'payment'],
            'education': ['student', 'school', 'university', 'grade', 'teacher', 'course', 'learning', 'academic'],
            'social_media': ['post', 'user', 'engagement', 'likes', 'platform', 'social', 'follower', 'content']
        }
        
        for domain, keywords in mappings.items():
            if any(k in details for k in keywords):
                return domain
        
        return 'generic'

    @staticmethod
    def get_schema(domain, project_details='', num_features=8):
        if domain in DomainSchemaInferrer.DOMAIN_SCHEMAS:
            return DomainSchemaInferrer.DOMAIN_SCHEMAS[domain]
        
        # Best effort extraction for 'generic' or any unknown domain
        if project_details:
            # Simple heuristic: find capitalized words or words before/after common connectors
            # This is a fallback to make it feel 'AI-like' for any project type
            words = re.findall(r'\b[A-Za-z]+\b', project_details)
            potential_cols = []
            seen = set()
            
            # Filter out common stop words
            stops = {'a', 'an', 'the', 'and', 'or', 'for', 'with', 'on', 'at', 'is', 'in', 'to', 'of'}
            
            for w in words:
                w_low = w.lower()
                if w_low not in stops and len(w) > 2 and w_low not in seen:
                    potential_cols.append(w.capitalize())
                    seen.add(w_low)
                if len(potential_cols) >= num_features:
                    break
            
            if potential_cols:
                schema = []
                for i, col in enumerate(potential_cols):
                    # Alternate between numerical and categorical
                    col_type = 'numerical' if i % 2 == 0 else 'categorical'
                    if col_type == 'numerical':
                        schema.append((col, 'numerical', {'mean': 100, 'std': 50}))
                    else:
                        schema.append((col, 'categorical', ['Low', 'Medium', 'High', 'Optimal']))
                return schema

        # Ultimate fallback
        return [(f'Metric_{i+1}', 'numerical', {'mean': 50, 'std': 20}) for i in range(num_features)]

class ManualDataGenerator:
    """Generate datasets based on user-defined specifications"""
    
    @staticmethod
    def generate(num_rows, columns_config):
        data = {}
        for col in columns_config:
            col_name = col['name']
            col_type = col['type']
            
            if col_type == 'numerical':
                data[col_name] = np.random.randn(num_rows) * 100 + 50
            elif col_type == 'categorical':
                categories = ['A', 'B', 'C', 'D', 'E']
                data[col_name] = np.random.choice(categories, num_rows)
        
        return pd.DataFrame(data)

class AutoDataGenerator:
    """Generate realistic datasets based on domain-aware schema inference"""

    @staticmethod
    def _generate_by_schema(num_rows, schema):
        data = {}
        for col_name, col_type, config in schema:
            if col_type == 'numerical':
                mean = config.get('mean', 50)
                std = config.get('std', 10)
                values = np.random.normal(mean, std, num_rows)
                
                if 'min' in config:
                    values = np.maximum(values, config['min'])
                if 'max' in config:
                    values = np.minimum(values, config['max'])
                
                # Round to 2 decimal places if it looks like currency or rating
                if any(k in col_name.lower() for k in ['price', 'amount', 'score', 'rating', 'value']):
                    data[col_name] = np.round(values, 2)
                else:
                    data[col_name] = np.round(values).astype(int)
                    
            elif col_type == 'categorical':
                data[col_name] = np.random.choice(config, num_rows)
        
        return pd.DataFrame(data)

    @staticmethod
    def generate_classification(num_rows, project_details=''):
        domain = DomainSchemaInferrer.infer_domain(project_details)
        schema = DomainSchemaInferrer.get_schema(domain, project_details=project_details)
        df = AutoDataGenerator._generate_by_schema(num_rows, schema)
        
        # Add a domain-specific target column
        target_name = 'target'
        if domain == 'ecommerce': target_name = 'will_purchase'
        elif domain == 'healthcare': target_name = 'readmitted'
        elif domain == 'finance': target_name = 'is_fraud'
        elif domain == 'education': target_name = 'will_pass'
        elif domain == 'social_media': target_name = 'is_viral'
        
        df[target_name] = np.random.choice([0, 1], num_rows, p=[0.7, 0.3])
        return df

    @staticmethod
    def generate_regression(num_rows, project_details=''):
        domain = DomainSchemaInferrer.infer_domain(project_details)
        schema = DomainSchemaInferrer.get_schema(domain, project_details=project_details)
        df = AutoDataGenerator._generate_by_schema(num_rows, schema)
        
        # Generate target with a simple linear relationship + noise
        target_name = 'label'
        if domain == 'ecommerce': target_name = 'lifetime_value'
        elif domain == 'healthcare': target_name = 'recovery_days'
        elif domain == 'finance': target_name = 'investment_return'
        elif domain == 'education': target_name = 'final_grade'
        elif domain == 'social_media': target_name = 'follower_growth'
        
        # Select first 3 numerical columns for base relationship
        num_cols = df.select_dtypes(include=[np.number]).columns[:3]
        if not num_cols.empty:
            df[target_name] = df[num_cols].sum(axis=1) * 1.5 + np.random.normal(0, 10, num_rows)
            df[target_name] = np.round(df[target_name], 2)
        else:
            df[target_name] = np.random.uniform(0, 100, num_rows)
            
        return df

    @staticmethod
    def generate_timeseries(num_rows, project_details=''):
        domain = DomainSchemaInferrer.infer_domain(project_details)
        df = pd.DataFrame()
        
        # Generate date range
        start_date = datetime.now() - timedelta(days=num_rows)
        df['timestamp'] = [start_date + timedelta(days=i) for i in range(num_rows)]
        
        # Domain specific trend values
        base_val = 100
        if domain == 'finance': base_val = 1000
        elif domain == 'social_media': base_val = 50000
        
        trend = np.linspace(base_val, base_val * 1.5, num_rows)
        seasonality = (base_val * 0.1) * np.sin(2 * np.pi * np.arange(num_rows) / 7)
        noise = np.random.randn(num_rows) * (base_val * 0.05)
        
        # Use a more descriptive column name based on domain
        val_name = 'metric_value'
        if domain == 'ecommerce': val_name = 'daily_revenue'
        elif domain == 'healthcare': val_name = 'patient_count'
        elif domain == 'finance': val_name = 'stock_price'
        elif domain == 'social_media': val_name = 'active_users'
        
        df[val_name] = np.round(trend + seasonality + noise, 2)
        return df

    @staticmethod
    def generate_clustering(num_rows, project_details=''):
        domain = DomainSchemaInferrer.infer_domain(project_details)
        schema = DomainSchemaInferrer.get_schema(domain, project_details=project_details)
        df = AutoDataGenerator._generate_by_schema(num_rows, schema)
        df['cluster_id'] = np.random.randint(0, 4, num_rows)
        return df

    @staticmethod
    def generate_recommendation(num_rows, project_details=''):
        domain = DomainSchemaInferrer.infer_domain(project_details)
        num_users = max(50, num_rows // 20)
        num_items = max(100, num_rows // 10)
        
        user_prefix = 'U'
        item_prefix = 'I'
        if domain == 'ecommerce': 
            user_prefix = 'CUST'
            item_prefix = 'PROD'
        elif domain == 'healthcare':
            user_prefix = 'PAT'
            item_prefix = 'DOC'
        elif domain == 'education':
            user_prefix = 'STU'
            item_prefix = 'CRSE'
            
        return pd.DataFrame({
            'user_id': [f'{user_prefix}{i:04d}' for i in np.random.randint(1, num_users + 1, num_rows)],
            'item_id': [f'{item_prefix}{i:05d}' for i in np.random.randint(1, num_items + 1, num_rows)],
            'rating': np.random.choice([1, 2, 3, 4, 5], num_rows, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
            'interaction_type': np.random.choice(['view', 'click', 'purchase', 'review'], num_rows),
            'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 8760)) for _ in range(num_rows)]
        })

    @staticmethod
    def generate(dataset_type, num_rows, project_details=''):
        original_type = dataset_type.lower()
        
        # Auto-detect task type if requested
        if original_type == 'auto' and project_details:
            details = project_details.lower()
            if any(k in details for k in ['time', 'series', 'date', 'temporal', 'historical', 'daily', 'monthly']):
                dataset_type = 'time-series'
            elif any(k in details for k in ['predict', 'sales', 'price', 'amount', 'value', 'growth', 'continuous']):
                dataset_type = 'regression'
            elif any(k in details for k in ['recommend', 'suggest', 'user', 'item', 'interaction', 'rating']):
                dataset_type = 'recommendation'
            elif any(k in details for k in ['group', 'cluster', 'segment', 'pattern']):
                dataset_type = 'clustering'
            else:
                # Default to classification for generic descriptions
                dataset_type = 'classification'
        elif original_type == 'auto':
            # Default fallback if no details provided
            dataset_type = 'classification'

        generators = {
            'classification': AutoDataGenerator.generate_classification,
            'regression': AutoDataGenerator.generate_regression,
            'time-series': AutoDataGenerator.generate_timeseries,
            'clustering': AutoDataGenerator.generate_clustering,
            'recommendation': AutoDataGenerator.generate_recommendation
        }
        
        generator = generators.get(dataset_type.lower())
        if generator:
            return generator(num_rows, project_details=project_details)
        return None
