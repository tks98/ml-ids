import pandas as pd
import numpy as np
import os
import logging
import pickle
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectFromModel
from collections import Counter

# Set up logging to track the progress and any issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to display a progress bar in the console
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    print(f'\r{prefix} |{bar}| {percents}% {suffix}', end='')

    if iteration == total:
        print()

# Function to load all CSV files from a directory into a single DataFrame
def load_all_csv_files(directory):
    """
    Load all CSV files from the specified directory into a single DataFrame.
    
    Args:
    directory (str): Path to the directory containing CSV files
    
    Returns:
    pandas.DataFrame: Combined DataFrame of all CSV files
    """
    # List to store individual DataFrames
    dataframes = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(directory, filename)
            
            # Load CSV file into a DataFrame
            df = pd.read_csv(file_path, low_memory=False)
            
            # Add a column to identify the source file
            df['source_file'] = filename
            
            # Append to the list of DataFrames
            dataframes.append(df)
            
            print(f"Loaded: {filename}")
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

# Function to load and preprocess the data, using caching for efficiency
def load_and_preprocess_data(directory, cache_file='full_dataset_cache.pkl'):
    # Check if processed data is already cached
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # If not cached, load and process the data
    logging.info("Cache not found. Loading all CSV files...")
    full_dataset = load_all_csv_files(directory)
    
    logging.info("Preprocessing data...")
    # Clean up column names by removing any leading/trailing whitespace
    full_dataset.columns = full_dataset.columns.str.strip()
    
    # Cache the processed data for future use
    logging.info("Saving data to cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump(full_dataset, f)
    
    return full_dataset

# Function to prepare features (X) and target (y) for machine learning
def prepare_features_and_target(df, sample_size=0.1, min_samples=10):
    # Sample a subset of the data to reduce processing time
    df_sampled = df.sample(frac=sample_size, random_state=42)
    
    # Identify the columns for the label and source file
    label_col = df_sampled.columns[df_sampled.columns.str.strip() == 'Label'][0]
    source_file_col = df_sampled.columns[df_sampled.columns.str.strip() == 'source_file'][0]
    
    # Print unique values in the Label column
    print("Unique attack types in the dataset:")
    print(df[label_col].unique())
    
    # Remove classes with too few samples to ensure reliable learning
    class_counts = df_sampled[label_col].value_counts()
    classes_to_keep = class_counts[class_counts >= min_samples].index
    df_sampled = df_sampled[df_sampled[label_col].isin(classes_to_keep)]
    
    # Separate features (X) and target (y)
    X = df_sampled.drop([label_col, source_file_col], axis=1)
    y = df_sampled[label_col]
    
    # Convert categorical variables to numerical (one-hot encoding)
    X = pd.get_dummies(X)
    # Replace infinite values with NaN and drop rows with NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    # Ensure y matches the rows in X after dropping NaN values
    y = y[X.index]
    
    # Add feature selection step
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    selector.fit(X, y)
    
    # Get the selected feature names
    selected_features = X.columns[selector.get_support()]
    
    # Apply the transformation
    X_selected = selector.transform(X)
    
    # Convert X_selected back to a DataFrame with the correct feature names
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    return X_selected, y

# Function to train and evaluate the machine learning model
def train_and_evaluate_model(X, y):
    logging.info("Starting model training and evaluation process...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Calculate class distribution
    class_counts = Counter(y_train)
    logging.info(f"Class distribution before SMOTE: {class_counts}")
    
    # Define sampling strategy
    sampling_strategy = {
        'Web Attack Brute Force': int(class_counts['BENIGN'] * 0.1),  # 10% of majority class
        'Web Attack XSS': int(class_counts['BENIGN'] * 0.1),  # 10% of majority class
        'Bot': int(class_counts['BENIGN'] * 0.05),  # 5% of majority class
    }
    
    # For other minority classes, oversample to 1% of majority class
    for class_name, count in class_counts.items():
        if count < class_counts['BENIGN'] and class_name not in sampling_strategy:
            sampling_strategy[class_name] = max(count, int(class_counts['BENIGN'] * 0.01))
    
    # Remove any classes from sampling_strategy that are not in the dataset
    sampling_strategy = {k: v for k, v in sampling_strategy.items() if k in class_counts}
    
    # Add Isolation Forest for anomaly detection
    logging.info("Training Isolation Forest for anomaly detection...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    anomaly_labels = iso_forest.fit_predict(X)
    
    # Identify anomalies
    anomalies = X[anomaly_labels == -1]
    logging.info(f"Number of anomalies detected: {len(anomalies)}")

    # Create a pipeline that scales the data, applies aggressive SMOTE, and then uses a Random Forest classifier
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(sampling_strategy=sampling_strategy, random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    # Train the model
    logging.info("Training model...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Calculate class distribution after SMOTE
    y_resampled = pipeline.named_steps['smote'].fit_resample(X_train, y_train)[1]
    class_counts_after = Counter(y_resampled)
    logging.info(f"Class distribution after SMOTE: {class_counts_after}")
    
    # Make predictions on the test set
    logging.info("Making predictions on test set...")
    y_pred = pipeline.predict(X_test)
    
    # Generate a classification report
    logging.info("Calculating classification report...")
    report = classification_report(y_test, y_pred, zero_division=1)
    print("\nClassification Report:")
    print(report)
    
    logging.info("Model evaluation completed.")
    
    return pipeline, X_test, y_test, y_pred, anomalies

# Function to plot a confusion matrix of model predictions
def plot_confusion_matrix(y_test, y_pred, top_n=10):
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    
    if len(unique_labels) > top_n:
        logging.info(f"Too many classes ({len(unique_labels)}). Limiting to top {top_n} most frequent.")
        label_counts = pd.Series(y_test).value_counts()
        top_labels = label_counts.nlargest(top_n).index
        mask = np.isin(y_test, top_labels) & np.isin(y_pred, top_labels)
        cm = confusion_matrix(y_test[mask], y_pred[mask])
        unique_labels = top_labels
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Function to plot feature importance from the Random Forest model
def plot_feature_importance(pipeline, X):
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with named columns")
    
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 10))
    feature_importances[:20].plot(kind='bar')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    logging.info("Feature importance plot saved as 'feature_importance.png'")

# Main function to orchestrate the entire process
def main():
    try:
        # Specify the directory containing the data files
        directory = 'data/MachineLearningCVE'
        # Load and preprocess the data
        full_dataset = load_and_preprocess_data(directory)
        
        # Prepare features and target for machine learning
        X, y = prepare_features_and_target(full_dataset)
        
        # Train and evaluate the model
        pipeline, X_test, y_test, y_pred, anomalies = train_and_evaluate_model(X, y)
        
        # Generate and save visualizations
        plot_confusion_matrix(y_test, y_pred)
        logging.info("Confusion matrix saved as 'confusion_matrix.png'")
        
        plot_feature_importance(pipeline, X)
        logging.info("Feature importance plot saved as 'feature_importance.png'")
        
        # Save anomalies to a CSV file
        anomalies.to_csv('anomalies.csv', index=False)
        logging.info("Anomalies saved to 'anomalies.csv'")

        logging.info("Process completed!")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    
    finally:
        logging.info("Script execution finished.")

# Entry point of the script
if __name__ == "__main__":
    main()