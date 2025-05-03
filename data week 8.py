import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

class DataAnalyzer:
    """
    A class to load, analyze, and visualize datasets using Pandas and Matplotlib.
    """
    
    def __init__(self, dataset_name='iris'):
        """
        Initialize the DataAnalyzer with a dataset.
        
        Args:
            dataset_name (str): Name of the dataset to load ('iris' by default)
        """
        self.dataset_name = dataset_name
        self.data = None
        self.load_data()
        
    def load_data(self):
        """
        Load the dataset based on the dataset_name.
        Handles different dataset sources and potential errors.
        """
        try:
            if self.dataset_name == 'iris':
                iris = load_iris()
                self.data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                       columns=iris['feature_names'] + ['target'])
                self.data['species'] = self.data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            else:
                # For other datasets, you would add more conditions here
                raise ValueError(f"Dataset '{self.dataset_name}' not supported.")
                
            print("Dataset loaded successfully!")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            self.data = None
            
    def explore_data(self):
        """
        Explore the dataset by displaying basic information.
        """
        if self.data is None:
            print("No data loaded to explore.")
            return
            
        print("\n=== Dataset Overview ===")
        print(f"Shape: {self.data.shape}")
        
        print("\n=== First 5 Rows ===")
        print(self.data.head())
        
        print("\n=== Data Types ===")
        print(self.data.dtypes)
        
        print("\n=== Missing Values ===")
        print(self.data.isnull().sum())
        
        print("\n=== Basic Statistics ===")
        print(self.data.describe())
        
    def clean_data(self):
        """
        Clean the dataset by handling missing values and incorrect data types.
        """
        if self.data is None:
            print("No data loaded to clean.")
            return
            
        print("\n=== Cleaning Data ===")
        
        # Handle missing values
        if self.data.isnull().sum().sum() > 0:
            print(f"Found {self.data.isnull().sum().sum()} missing values.")
            # For numerical columns, fill with median
            num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            for col in num_cols:
                if self.data[col].isnull().sum() > 0:
                    median_val = self.data[col].median()
                    self.data[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            cat_cols = self.data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if self.data[col].isnull().sum() > 0:
                    mode_val = self.data[col].mode()[0]
                    self.data[col].fillna(mode_val, inplace=True)
                    print(f"Filled missing values in {col} with mode: {mode_val}")
        else:
            print("No missing values found.")
            
        print("Data cleaning completed.")
        
    def analyze_data(self):
        """
        Perform basic data analysis including grouping and aggregations.
        """
        if self.data is None:
            print("No data loaded to analyze.")
            return
            
        print("\n=== Data Analysis ===")
        
        # Group by species and calculate mean for numerical columns
        if 'species' in self.data.columns:
            print("\nMean values by species:")
            print(self.data.groupby('species').mean())
            
        # Correlation analysis
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            print("\nCorrelation matrix:")
            print(self.data[num_cols].corr())
            
    def visualize_data(self):
        """
        Create various visualizations to explore the data.
        """
        if self.data is None:
            print("No data loaded to visualize.")
            return
            
        print("\nCreating visualizations...")
        plt.figure(figsize=(15, 10))
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Line chart (simulating time series by using index as x-axis)
        plt.subplot(2, 2, 1)
        if 'sepal length (cm)' in self.data.columns:
            self.data['sepal length (cm)'].plot(kind='line', color='green')
            plt.title('Sepal Length Trend (by index)')
            plt.xlabel('Index')
            plt.ylabel('Sepal Length (cm)')
        else:
            plt.text(0.5, 0.5, 'No suitable column for line chart', ha='center')
            
        # 2. Bar chart (average measurement by species)
        plt.subplot(2, 2, 2)
        if 'species' in self.data.columns and 'petal length (cm)' in self.data.columns:
            self.data.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color=['blue', 'orange', 'green'])
            plt.title('Average Petal Length by Species')
            plt.xlabel('Species')
            plt.ylabel('Petal Length (cm)')
        else:
            plt.text(0.5, 0.5, 'No suitable columns for bar chart', ha='center')
            
        # 3. Histogram (distribution of sepal width)
        plt.subplot(2, 2, 3)
        if 'sepal width (cm)' in self.data.columns:
            sns.histplot(self.data['sepal width (cm)'], bins=15, kde=True, color='purple')
            plt.title('Distribution of Sepal Width')
            plt.xlabel('Sepal Width (cm)')
            plt.ylabel('Frequency')
        else:
            plt.text(0.5, 0.5, 'No suitable column for histogram', ha='center')
            
        # 4. Scatter plot (sepal length vs petal length)
        plt.subplot(2, 2, 4)
        if 'sepal length (cm)' in self.data.columns and 'petal length (cm)' in self.data.columns:
            sns.scatterplot(data=self.data, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='viridis')
            plt.title('Sepal Length vs Petal Length')
            plt.xlabel('Sepal Length (cm)')
            plt.ylabel('Petal Length (cm)')
            plt.legend(title='Species')
        else:
            plt.text(0.5, 0.5, 'No suitable columns for scatter plot', ha='center')
            
        plt.tight_layout()
        plt.show()
        
    def run_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print(f"\n{'='*50}")
        print(f"Analyzing {self.dataset_name} dataset")
        print(f"{'='*50}")
        
        self.explore_data()
        self.clean_data()
        self.analyze_data()
        self.visualize_data()
        
        print("\nAnalysis complete!")


# Main execution
if __name__ == "__main__":
    # Create an instance of the DataAnalyzer
    analyzer = DataAnalyzer('iris')
    
    # Run the complete analysis
    analyzer.run_analysis()