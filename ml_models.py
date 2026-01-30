"""
Machine Learning Models and Datasets Module
Provides pre-trained models and sample datasets for the XplainAI platform
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LoanApprovalDataset:
    """Loan Approval Dataset - Simulated banking scenario"""
    
    @staticmethod
    def get_data():
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic loan application data
        credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
        income = np.random.normal(60000, 25000, n_samples).clip(15000, 200000)
        loan_amount = np.random.normal(150000, 75000, n_samples).clip(10000, 500000)
        employment_years = np.random.exponential(5, n_samples).clip(0, 40)
        debt_to_income = (np.random.beta(2, 5, n_samples) * 0.6).clip(0, 0.8)
        
        # Generate approval based on realistic criteria
        approval_score = (
            (credit_score - 300) / 550 * 0.35 +
            (income / 200000) * 0.25 +
            (1 - loan_amount / 500000) * 0.20 +
            (employment_years / 40) * 0.10 +
            (1 - debt_to_income / 0.8) * 0.10
        )
        
        noise = np.random.normal(0, 0.1, n_samples)
        approval = (approval_score + noise > 0.5).astype(int)
        
        df = pd.DataFrame({
            'credit_score': credit_score.round(0),
            'annual_income': income.round(0),
            'loan_amount': loan_amount.round(0),
            'employment_years': employment_years.round(1),
            'debt_to_income_ratio': debt_to_income.round(3),
            'approved': approval
        })
        
        return df
    
    @staticmethod
    def get_feature_names():
        return ['credit_score', 'annual_income', 'loan_amount', 'employment_years', 'debt_to_income_ratio']
    
    @staticmethod
    def get_feature_descriptions():
        return {
            'credit_score': 'Credit Score (300-850)',
            'annual_income': 'Annual Income ($)',
            'loan_amount': 'Requested Loan Amount ($)',
            'employment_years': 'Years of Employment',
            'debt_to_income_ratio': 'Debt-to-Income Ratio (0-1)'
        }
    
    @staticmethod
    def get_input_ranges():
        return {
            'credit_score': (300, 850, 650),
            'annual_income': (15000, 200000, 60000),
            'loan_amount': (10000, 500000, 150000),
            'employment_years': (0, 40, 5),
            'debt_to_income_ratio': (0.0, 0.8, 0.3)
        }


class StudentAdmissionDataset:
    """Student Admission Dataset - University admission scenario"""
    
    @staticmethod
    def get_data():
        np.random.seed(42)
        n_samples = 800
        
        # Generate realistic student data
        gpa = np.random.normal(3.3, 0.5, n_samples).clip(2.0, 4.0)
        test_score = np.random.normal(1200, 200, n_samples).clip(800, 1600)
        extracurricular = np.random.poisson(3, n_samples).clip(0, 10)
        essay_score = np.random.normal(7.5, 1.5, n_samples).clip(1, 10)
        
        # Generate admission based on criteria
        admission_score = (
            (gpa - 2.0) / 2.0 * 0.35 +
            (test_score - 800) / 800 * 0.30 +
            (extracurricular / 10) * 0.15 +
            (essay_score / 10) * 0.20
        )
        
        noise = np.random.normal(0, 0.12, n_samples)
        admitted = (admission_score + noise > 0.5).astype(int)
        
        df = pd.DataFrame({
            'gpa': gpa.round(2),
            'test_score': test_score.round(0),
            'extracurricular_count': extracurricular,
            'essay_score': essay_score.round(1),
            'admitted': admitted
        })
        
        return df
    
    @staticmethod
    def get_feature_names():
        return ['gpa', 'test_score', 'extracurricular_count', 'essay_score']
    
    @staticmethod
    def get_feature_descriptions():
        return {
            'gpa': 'Grade Point Average (2.0-4.0)',
            'test_score': 'Standardized Test Score (800-1600)',
            'extracurricular_count': 'Number of Extracurricular Activities',
            'essay_score': 'Essay Quality Score (1-10)'
        }
    
    @staticmethod
    def get_input_ranges():
        return {
            'gpa': (2.0, 4.0, 3.3),
            'test_score': (800, 1600, 1200),
            'extracurricular_count': (0, 10, 3),
            'essay_score': (1.0, 10.0, 7.5)
        }


class MLModelManager:
    """Manages ML models for different datasets"""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if dataset_name == "Loan Approval":
            self.dataset = LoanApprovalDataset()
        elif dataset_name == "Student Admission":
            self.dataset = StudentAdmissionDataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def train_models(self):
        """Train all available models"""
        df = self.dataset.get_data()
        feature_names = self.dataset.get_feature_names()
        
        X = df[feature_names].values
        y = df.iloc[:, -1].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Decision Tree
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        dt_accuracy = dt.score(X_test, y_test)
        
        # Train Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_accuracy = lr.score(X_test_scaled, y_test)
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        knn_accuracy = knn.score(X_test_scaled, y_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        rf_accuracy = rf.score(X_test, y_test)
        
        self.models = {
            'Decision Tree': {'model': dt, 'accuracy': dt_accuracy, 'needs_scaling': False},
            'Logistic Regression': {'model': lr, 'accuracy': lr_accuracy, 'needs_scaling': True},
            'KNN': {'model': knn, 'accuracy': knn_accuracy, 'needs_scaling': True},
            'Random Forest': {'model': rf, 'accuracy': rf_accuracy, 'needs_scaling': False}
        }
        
        self.is_trained = True
        return self.models
    
    def predict(self, model_name, input_data):
        """Make prediction with specified model"""
        if not self.is_trained:
            self.train_models()
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Prepare input
        input_array = np.array(input_data).reshape(1, -1)
        
        if model_info['needs_scaling']:
            input_array = self.scaler.transform(input_array)
        
        prediction = model.predict(input_array)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_array)[0]
            confidence = proba[prediction]
        else:
            confidence = None
        
        return prediction, confidence
    
    def get_feature_importance(self, model_name):
        """Get feature importance for the model"""
        if not self.is_trained:
            self.train_models()
        
        model = self.models[model_name]['model']
        feature_names = self.dataset.get_feature_names()
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        # Normalize importance
        importance = importance / importance.sum()
        
        return dict(zip(feature_names, importance))
    
    def explain_decision(self, model_name, input_data, prediction):
        """Generate human-readable explanation"""
        feature_names = self.dataset.get_feature_names()
        feature_descriptions = self.dataset.get_feature_descriptions()
        importance = self.get_feature_importance(model_name)
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        explanations = []
        
        if self.dataset_name == "Loan Approval":
            if prediction == 1:
                explanations.append("✅ **Loan Application Approved**")
                explanations.append("\n**Key Positive Factors:**")
            else:
                explanations.append("❌ **Loan Application Rejected**")
                explanations.append("\n**Areas of Concern:**")
            
            # Credit Score
            credit_score = input_data[0]
            if credit_score >= 700:
                explanations.append(f"• Excellent credit score ({credit_score:.0f}) shows financial responsibility")
            elif credit_score >= 650:
                explanations.append(f"• Good credit score ({credit_score:.0f}) demonstrates reliability")
            elif credit_score >= 600:
                explanations.append(f"• Fair credit score ({credit_score:.0f}) - consider improving credit history")
            else:
                explanations.append(f"• Low credit score ({credit_score:.0f}) - this is a major concern")
            
            # Income
            income = input_data[1]
            loan_amount = input_data[2]
            if income >= loan_amount * 0.3:
                explanations.append(f"• Strong income (${income:,.0f}) relative to loan amount")
            else:
                explanations.append(f"• Income (${income:,.0f}) may be low for requested loan amount")
            
            # Debt-to-Income
            dti = input_data[4]
            if dti <= 0.35:
                explanations.append(f"• Healthy debt-to-income ratio ({dti:.1%})")
            elif dti <= 0.45:
                explanations.append(f"• Moderate debt-to-income ratio ({dti:.1%})")
            else:
                explanations.append(f"• High debt-to-income ratio ({dti:.1%}) - consider reducing debt")
            
            # Employment
            employment = input_data[3]
            if employment >= 5:
                explanations.append(f"• Stable employment history ({employment:.1f} years)")
            elif employment >= 2:
                explanations.append(f"• Moderate employment stability ({employment:.1f} years)")
            else:
                explanations.append(f"• Limited employment history ({employment:.1f} years)")
        
        elif self.dataset_name == "Student Admission":
            if prediction == 1:
                explanations.append("🎓 **Admission Approved - Welcome!**")
                explanations.append("\n**Your Strengths:**")
            else:
                explanations.append("📋 **Admission Denied**")
                explanations.append("\n**Areas to Improve:**")
            
            # GPA
            gpa = input_data[0]
            if gpa >= 3.5:
                explanations.append(f"• Outstanding GPA ({gpa:.2f}) - exceptional academic performance")
            elif gpa >= 3.0:
                explanations.append(f"• Good GPA ({gpa:.2f}) - solid academic foundation")
            elif gpa >= 2.5:
                explanations.append(f"• Fair GPA ({gpa:.2f}) - consider improving academic performance")
            else:
                explanations.append(f"• Low GPA ({gpa:.2f}) - this significantly impacts admission")
            
            # Test Score
            test_score = input_data[1]
            if test_score >= 1400:
                explanations.append(f"• Excellent test score ({test_score:.0f}) demonstrates strong aptitude")
            elif test_score >= 1200:
                explanations.append(f"• Good test score ({test_score:.0f}) shows capability")
            elif test_score >= 1000:
                explanations.append(f"• Average test score ({test_score:.0f}) - consider retaking")
            else:
                explanations.append(f"• Below average test score ({test_score:.0f}) needs improvement")
            
            # Extracurriculars
            extra = input_data[2]
            if extra >= 5:
                explanations.append(f"• Impressive involvement in {extra} extracurricular activities")
            elif extra >= 3:
                explanations.append(f"• Good participation in {extra} extracurricular activities")
            else:
                explanations.append(f"• Limited extracurricular involvement ({extra}) - consider expanding")
            
            # Essay
            essay = input_data[3]
            if essay >= 8:
                explanations.append(f"• Excellent essay score ({essay:.1f}/10) - compelling narrative")
            elif essay >= 6:
                explanations.append(f"• Good essay score ({essay:.1f}/10) - shows potential")
            else:
                explanations.append(f"• Essay score ({essay:.1f}/10) needs improvement")
        
        return "\n".join(explanations)


def get_available_datasets():
    """Return list of available datasets"""
    return ["Loan Approval", "Student Admission"]


def get_available_algorithms():
    """Return list of available algorithms"""
    return ["Decision Tree", "Logistic Regression", "KNN", "Random Forest"]


def get_algorithm_descriptions():
    """Return descriptions of algorithms"""
    return {
        'Decision Tree': 'Makes decisions using a tree-like structure. Easy to understand and visualize.',
        'Logistic Regression': 'Uses statistical relationships between features. Fast and reliable.',
        'KNN': 'Predicts based on similar past cases. Intuitive "nearest neighbor" approach.',
        'Random Forest': 'Combines multiple decision trees for robust predictions. High accuracy.'
    }
