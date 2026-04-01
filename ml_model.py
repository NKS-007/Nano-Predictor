import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ScientificDataAnalyzer:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.problem_type = None
        self.target_column = None
        
    def load_data(self, csv_file):
        """Load CSV data and perform initial analysis"""
        try:
            self.data = pd.read_csv(csv_file)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            print("\nDataset Info:")
            print(self.data.info())
            print("\nFirst 5 rows:")
            print(self.data.head())
            print("\nBasic Statistics:")
            print(self.data.describe())
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Missing values
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        print(missing[missing > 0])
        
        # Data types
        print(f"\nData Types:")
        print(self.data.dtypes)
        
        # Correlation matrix for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
        
        # Distribution plots
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:4]):
                if i < len(axes):
                    self.data[col].hist(bins=30, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
    
    def preprocess_data(self, target_column, test_size=0.2):
        """Preprocess data for machine learning"""
        if self.data is None:
            print("Please load data first!")
            return False
        
        if target_column not in self.data.columns:
            print(f"Target column '{target_column}' not found in dataset!")
            return False
        
        self.target_column = target_column
        
        # Separate features and target
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        
        # Handle categorical variables
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        # Handle missing values
        self.X = self.X.fillna(self.X.mean())
        
        # Determine problem type
        if self.y.dtype == 'object' or len(self.y.unique()) < 10:
            self.problem_type = 'classification'
            if self.y.dtype == 'object':
                le = LabelEncoder()
                self.y = le.fit_transform(self.y)
        else:
            self.problem_type = 'regression'
        
        print(f"Problem Type: {self.problem_type}")
        print(f"Features: {list(self.X.columns)}")
        print(f"Target: {target_column}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return True
    
    def train_models(self):
        """Train multiple models and compare performance"""
        if self.X_train is None:
            print("Please preprocess data first!")
            return
        
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        if self.problem_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf')
            }
            scoring = 'r2'
        else:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVC': SVC(kernel='rbf', random_state=42)
            }
            scoring = 'accuracy'
        
        best_score = -np.inf
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for SVM and Logistic Regression
            if name in ['SVR', 'SVC', 'Logistic Regression']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring=scoring)
            
            # Predictions
            y_pred = model.predict(X_test_use)
            
            # Evaluate
            if self.problem_type == 'regression':
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                print(f"  Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                print(f"  Test R²: {r2:.4f}")
                print(f"  Test MSE: {mse:.4f}")
                score = r2
            else:
                acc = accuracy_score(self.y_test, y_pred)
                print(f"  Cross-validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                print(f"  Test Accuracy: {acc:.4f}")
                score = acc
            
            # Store model
            self.models[name] = {
                'model': model,
                'score': score,
                'cv_scores': cv_scores,
                'scaled': name in ['SVR', 'SVC', 'Logistic Regression']
            }
            
            # Track best model
            if score > best_score:
                best_score = score
                self.best_model = name
        
        print(f"\nBest Model: {self.best_model} with score: {best_score:.4f}")
    
    def feature_importance(self):
        """Display feature importance for tree-based models"""
        if self.best_model and 'Random Forest' in self.best_model:
            model = self.models[self.best_model]['model']
            importances = model.feature_importances_
            feature_names = self.X.columns
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1]
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
            
            print("\nFeature Importance Rankings:")
            for i in indices:
                print(f"{feature_names[i]}: {importances[i]:.4f}")
    
    def make_predictions(self, new_data=None):
        """Make predictions on new data"""
        if self.best_model is None:
            print("Please train models first!")
            return None
        
        if new_data is None:
            # Use test set
            if self.models[self.best_model]['scaled']:
                X_pred = self.X_test_scaled
            else:
                X_pred = self.X_test
            actual = self.y_test
        else:
            # Preprocess new data same way as training data
            X_pred = new_data.copy()
            categorical_cols = X_pred.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X_pred[col] = le.fit_transform(X_pred[col].astype(str))
            X_pred = X_pred.fillna(X_pred.mean())
            
            if self.models[self.best_model]['scaled']:
                X_pred = self.scaler.transform(X_pred)
            
            actual = None
        
        model = self.models[self.best_model]['model']
        predictions = model.predict(X_pred)
        
        if actual is not None:
            # Show some example predictions vs actual
            comparison = pd.DataFrame({
                'Actual': actual[:10],
                'Predicted': predictions[:10],
                'Difference': actual[:10] - predictions[:10]
            })
            print("\nSample Predictions vs Actual:")
            print(comparison)
        
        return predictions
    
    def save_model(self, filename='scientific_model.pkl'):
        """Save the best model"""
        if self.best_model:
            model_data = {
                'model': self.models[self.best_model]['model'],
                'scaler': self.scaler,
                'feature_names': list(self.X.columns),
                'problem_type': self.problem_type,
                'scaled': self.models[self.best_model]['scaled']
            }
            joblib.dump(model_data, filename)
            print(f"Model saved as {filename}")
    
    def load_model(self, filename='scientific_model.pkl'):
        """Load a saved model"""
        try:
            model_data = joblib.load(filename)
            self.best_model = 'Loaded Model'
            self.models[self.best_model] = {
                'model': model_data['model'],
                'scaled': model_data['scaled']
            }
            self.scaler = model_data['scaler']
            self.problem_type = model_data['problem_type']
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

# Example usage and demonstration
def main():
    print("="*60)
    print("SCIENTIFIC DATA MACHINE LEARNING ANALYZER")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ScientificDataAnalyzer()
    
    # Example with synthetic scientific data
    print("Creating sample scientific dataset...")
    
    # Create sample dataset (you can replace this with your CSV file)
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(25, 5, n_samples),
        'pressure': np.random.normal(1013, 50, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'ph_level': np.random.normal(7, 1, n_samples),
        'concentration': np.random.uniform(0.1, 2.0, n_samples),
    })
    
    # Create target variable (reaction_rate) based on other variables
    sample_data['reaction_rate'] = (
        0.3 * sample_data['temperature'] + 
        0.1 * sample_data['pressure'] + 
        0.2 * sample_data['humidity'] + 
        0.4 * sample_data['ph_level'] + 
        0.5 * sample_data['concentration'] +
        np.random.normal(0, 2, n_samples)
    )
    
    # Save sample data
    sample_data.to_csv('Data/data.csv', index=False)
    print("Sample data saved as 'scientific_sample_data.csv'")
    
    # Load and analyze data
    if analyzer.load_data('scientific_sample_data.csv'):
        # Perform EDA
        analyzer.exploratory_data_analysis()
        
        # Preprocess data
        if analyzer.preprocess_data('reaction_rate'):
            # Train models
            analyzer.train_models()
            
            # Show feature importance
            analyzer.feature_importance()
            
            # Make predictions
            predictions = analyzer.make_predictions()
            
            # Save model
            analyzer.save_model()

if __name__ == "__main__":
    main()