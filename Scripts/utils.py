import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class OverviewAnalysis:
    def __init__(self, df):
        self.df = df

    def dataset(self):
        dataset_data = self.df
        return dataset_data

    def handle_missing_values(self,df):
        """ Handle missing values by replacing numeric columns with the mean and categorical columns with the mode """
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = self.df.select_dtypes(exclude=['float64', 'int64']).columns

        # Fill numeric columns with mean
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Fill non-numeric columns with mode
        self.df[non_numeric_cols] = self.df[non_numeric_cols].fillna(self.df[non_numeric_cols].mode().iloc[0])
        print("Missing values handled.")
        # return self.df[numeric_cols] , self.df[non_numeric_cols]

       

    def handle_outliers(self):
        """ Handle outliers using the IQR method """
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64'])

        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1

        # Filter out outliers
        self.df = self.df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
        print("Outliers handled.")
        # return self.df
    def create_decile_classes(self):
        """ Create decile classes based on total session duration """
        self.df['Total Duration'] = self.df['Dur. (ms)']
        self.df['Total Data'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']
        self.df['Decile Class'] = pd.qcut(self.df['Total Duration'], 10, labels=False, duplicates='drop')
        # print("Decile classes created.")
        Decline_data = self.df.groupby('Decile Class')['Total Data'].sum()
        print("Decline Data", Decline_data)
        return self.df['Total Duration'],self.df['Total Data'],self.df['Decile Class']

    def basic_metrics(self):
        """ Calculate and print basic metrics (mean, median, etc.) for total duration and data """
        basic_metrics = self.df[['Total Duration', 'Total DL (Bytes)', 'Total UL (Bytes)']].describe()
        print("Basic Metrics:\n", basic_metrics)

    def univariate_analysis(self):
        """ Perform non-graphical univariate analysis to calculate dispersion parameters """
        dispersion_params = self.df[['Total Duration', 'Total DL (Bytes)', 'Total UL (Bytes)']].agg(['var', 'std', 'min', 'max'])
        print("Dispersion Parameters:\n", dispersion_params)
        return dispersion_params

    def graphical_univariate_analysis(self):
        """ Perform graphical univariate analysis """
        sns.histplot(self.df['Total Duration'])
        plt.title('Distribution of Total Session Duration')
        plt.show()

        sns.boxplot(x=self.df['Total Data'])
        plt.title('Boxplot of Total Data')
        plt.show()

    def bivariate_analysis(self):
        """ Perform bivariate analysis between each application and total data """
        app_cols = ['Youtube DL (Bytes)', 'Social Media DL (Bytes)', 'Netflix DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Gaming DL (Bytes)']
        self.df['Total Data'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']

        for app in app_cols:
            sns.scatterplot(x=self.df[app], y=self.df['Total Data'])
            plt.title(f'{app} vs Total Data')
            plt.show()

    def correlation_analysis(self):
        """ Compute and plot correlation matrix for app-related data """
        app_cols = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        correlation_matrix = self.df[app_cols].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix Between Application Data Usage')
        plt.show()

        print("Correlation Matrix:\n", correlation_matrix)

    def perform_pca(self):
        """ Perform dimensionality reduction using Principal Component Analysis (PCA) """
        app_cols = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        scaler = StandardScaler()

        # Scale the data
        scaled_data = scaler.fit_transform(self.df[app_cols])

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        explained_variance = pca.explained_variance_ratio_
        print("Explained variance by PCA components:", explained_variance)

# Usage:
# Load your data
# df = pd.read_excel('/mnt/data/datasets.xlsx')

# Initialize the class
# analysis = TelecomDataAnalysis(df)

# Call the methods to perform various tasks
# analysis.handle_missing_values()
# analysis.handle_outliers()
# analysis.create_decile_classes()
# analysis.basic_metrics()
# analysis.univariate_analysis()
# analysis.graphical_univariate_analysis()
# analysis.bivariate_analysis()
# analysis.correlation_analysis()
# analysis.perform_pca()
