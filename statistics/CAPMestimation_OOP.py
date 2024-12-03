# Step 1: Install Necessary Packages, if necessary, from terminal by
#           pip install pandas statsmodels matplotlib seaborn openpyxl

# Step 2: Import the Packages in Python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("All libraries imported successfully.")

# Step 2.1: Choose to analyze either one or all portfolios
analysis_mode = input("Enter '1' to select a particular portfolio, '2' to consider all portfolios: ")

# Step 3: Class to Load the Data
class DataLoader:
    def __init__(self, file_location, sheet_name='Data'):
        self.file_location = file_location
        self.sheet_name = sheet_name
        self.dataset = None

    def load_data(self):
        try:
            excel_file = pd.ExcelFile(self.file_location)
            print("Available sheets:", excel_file.sheet_names)
            self.dataset = pd.read_excel(self.file_location, sheet_name=self.sheet_name, header=4)
            print("Data loaded successfully from sheet:", self.sheet_name)
        except FileNotFoundError:
            print(f"File not found: {self.file_location}")
            raise
        except PermissionError:
            print(f"Permission denied: {self.file_location}")
            raise
        except Exception as e:
            print(f"An error occurred while loading the Excel file: {e}")
            raise

        self.dataset.columns = self.dataset.columns.str.strip()
        print("Column names:", self.dataset.columns.tolist())
        return self.dataset

# Step 4: Class to Preprocess the Data
class DataProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.excess_return_data = None

    def preprocess(self):
        try:
            self.dataset['Date'] = pd.to_datetime(self.dataset['Date'], format='%Y%m%d')
            self.dataset['RF'] = self.dataset['RF'].astype(float)
            self.dataset['Mkt-RF'] = self.dataset['Mkt-RF'].astype(float)
            self.excess_return_data = self.dataset.iloc[:, 7:17].subtract(self.dataset['RF'], axis=0).astype(float)
            print("Data preprocessed successfully.")
        except KeyError as e:
            print(f"Key error: {e}")
            print("Please check if the column names are correct in your Excel file.")
            raise
        except Exception as e:
            print(f"An error occurred during data preprocessing: {e}")
            raise
        return self.excess_return_data

# Step 5: Class to Perform OLS Regression for chosen portfolio(s)
class RegressionAnalyzer:
    def __init__(self, dataset, excess_return_data):
        self.dataset = dataset
        self.excess_return_data = excess_return_data
        self.analysis_results = {}

    def run_regression(self, mode):
        market_excess_return = self.dataset['Mkt-RF']
        X = sm.add_constant(market_excess_return)

        if mode == '1':
            selected_portfolio = input("Enter the name of the portfolio: ")
            y_portfolio = self.excess_return_data[selected_portfolio].astype(float)
            self.analysis_results[selected_portfolio] = sm.OLS(y_portfolio, X).fit()
        elif mode == '2':
            for portafoglio in self.excess_return_data.columns:
                y_portfolio = self.excess_return_data[portafoglio].astype(float)
                self.analysis_results[portafoglio] = sm.OLS(y_portfolio, X).fit()
        print("OLS regression performed successfully.")
        return self.analysis_results

# Step 6: Class to Display and Interpret the Results
class Visualizer:
    def __init__(self, analysis_results, excess_return_data):
        self.analysis_results = analysis_results
        self.excess_return_data = excess_return_data

    def plot_results(self, portafoglio, modello):
        fitted_values = modello.fittedvalues.astype(float)
        actual_values = self.excess_return_data[portafoglio].astype(float)
        plt.scatter(fitted_values, actual_values)
        plt.plot(fitted_values, modello.fittedvalues, color='red')
        intercept, slope = modello.params
        plt.title(f'{portafoglio} Portfolio')
        plt.text(0.05, 0.95, f'$\\alpha$: {intercept:.4f}\n$\\beta$: {slope:.4f}', transform=plt.gca().transAxes, verticalalignment='top')

    def generate_plots(self, mode):
        if mode == '1':
            for portafoglio, modello in self.analysis_results.items():
                plt.figure()
                self.plot_results(portafoglio, modello)
                plt.show()
        elif mode == '2':
            plt.figure(figsize=(15, 10))
            num_plots = len(self.analysis_results)
            num_cols = 5
            num_rows = (num_plots + num_cols - 1) // num_cols
            for i, (portafoglio, modello) in enumerate(self.analysis_results.items(), start=1):
                plt.subplot(num_rows, num_cols, i)
                self.plot_results(portafoglio, modello)
            plt.tight_layout()
            plt.show()
        print("Plots generated successfully.")

# Step 8: Class for Hypothesis Testing
class HypothesisTester:
    def __init__(self, CAPMmodel):
        self.model = CAPMmodel

    def run_test(self, alpha_null, beta_mkt_null, significance_level):
        r = np.zeros((2, len(self.model.params)))
        r[0, 0] = 1
        r[1, 1] = 1
        q = np.array([alpha_null, beta_mkt_null])
        f_test = self.model.f_test((r, q))
        print(f"Hypothesis test result: {f_test}")
        p_value = f_test.pvalue
        if p_value < significance_level:
            print(f"$H_0$ rejected at {significance_level*100}% level (p-value={p_value:.4e})")
        else:
            print(f"$H_0$ not rejected at {significance_level*100}% level (p-value={p_value:.4e})")

# Step 9: Class for Residual Analysis
class ResidualAnalyzer:
    def __init__(self, CAPMmodel):
        self.model = CAPMmodel

    def analyze(self):
        residuals = self.model.resid
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sns.scatterplot(x=self.model.fittedvalues, y=residuals, ax=axes[0])
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_title('Residual Plot')
        sm.qqplot(residuals, line='s', ax=axes[1])
        axes[1].set_title('Q-Q Plot')
        sns.histplot(residuals, kde=True, ax=axes[2])
        axes[2].set_title('Histogram of Residuals')
        plt.tight_layout()
        plt.show()

# Main Execution
if __name__ == "__main__":
    main_file_path = '/Users/sakshiii/Desktop/Intro_to_python/statistics/F-F_Research_Data_5_Factors_2x3_daily_August2024.xlsx'
    loader = DataLoader(main_file_path)
    main_dataset = loader.load_data()
    processor = DataProcessor(main_dataset)
    main_excess_returns = processor.preprocess()
    analyzer = RegressionAnalyzer(main_dataset, main_excess_returns)
    main_regression_results = analyzer.run_regression(analysis_mode)
    visualizer = Visualizer(main_regression_results, main_excess_returns)
    visualizer.generate_plots(analysis_mode)

    for portfolio, model in main_regression_results.items():
        print(f"Testing hypothesis for {portfolio}:")
        tester = HypothesisTester(model)
        tester.run_test(alpha_null=0, beta_mkt_null=1, significance_level=0.05)

        print(f"Residual analysis for {portfolio}:")
        residual_analyzer = ResidualAnalyzer(model)
        residual_analyzer.analyze()
