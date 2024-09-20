import pandas as pd
import numpy as np
import os
import logging
import pickle
import time
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel
from collections import Counter

# Set up logging to track the progress and any issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            
            # Load CSV file into a DataFrame with proper encoding
            df = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')
            
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
def prepare_features_and_target(df, sample_size=0.5, min_samples=10):
    # Sample a larger subset of the data
    df_sampled = df.sample(frac=sample_size, random_state=42)
    
    # Identify the columns for the label and source file
    label_col = df_sampled.columns[df_sampled.columns.str.strip() == 'Label'][0]
    source_file_col = df_sampled.columns[df_sampled.columns.str.strip() == 'source_file'][0]
    
    # Clean labels
    df_sampled[label_col] = df_sampled[label_col].str.replace('-', '', regex=False)  # Remove hyphens
    df_sampled[label_col] = df_sampled[label_col].str.strip()
    
    # Print unique values in the Label column
    print("Unique attack types in the dataset:")
    print(df_sampled[label_col].unique())
    
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
    y = y.loc[X.index]
    
    return X, y

# Function to train and evaluate the machine learning model
def train_and_evaluate_model(X, y, model_cache_file='trained_model_cache.joblib'):
    logging.info("Starting model training and evaluation process...")
    
    # Check if a cached model exists
    if os.path.exists(model_cache_file):
        logging.info("Loading model from cache...")
        pipeline = joblib.load(model_cache_file)
        
        # Split data into training and testing sets
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Calculate class distribution
        class_counts = Counter(y_train)
        logging.info(f"Class distribution in training set: {class_counts}")
        
        # Define the pipeline
        pipeline = ImbPipeline([
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
        ])
        
        # Train the model
        logging.info("Training model...")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Cache the trained model
        logging.info("Saving model to cache...")
        joblib.dump(pipeline, model_cache_file)
        
        # Calculate class distribution after SMOTE
        X_resampled, y_resampled = pipeline.named_steps['smote'].fit_resample(
            pipeline.named_steps['feature_selection'].transform(X_train),
            y_train
        )
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
    
    return pipeline, X_test, y_test, y_pred

# Function to plot a confusion matrix of model predictions
def plot_confusion_matrix(y_test, y_pred, top_n=10):
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    
    if len(unique_labels) > top_n:
        logging.info(f"Too many classes ({len(unique_labels)}). Limiting to top {top_n} most frequent.")
        label_counts = pd.Series(y_test).value_counts()
        top_labels = label_counts.nlargest(top_n).index
        mask = np.isin(y_test, top_labels)
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=top_labels)
    else:
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        top_labels = unique_labels
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=top_labels, yticklabels=top_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    logging.info("Confusion matrix saved as 'confusion_matrix.png'")

# Function to plot feature importance from the Random Forest model
def plot_feature_importance(pipeline, X):
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # Get the selected feature names from feature selection
    selector_support = pipeline.named_steps['feature_selection'].get_support()
    selected_features = X.columns[selector_support]
    
    feature_importances = pd.Series(importances, index=selected_features).sort_values(ascending=False)
    
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
        X, y = prepare_features_and_target(full_dataset, sample_size=0.5)
        
        # Train and evaluate the model
        pipeline, X_test, y_test, y_pred = train_and_evaluate_model(X, y)
        
        # Generate and save visualizations
        plot_confusion_matrix(y_test, y_pred)
        logging.info("Confusion matrix saved as 'confusion_matrix.png'")
        
        plot_feature_importance(pipeline, X)
        logging.info("Feature importance plot saved as 'feature_importance.png'")
        
        logging.info("Process completed!")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    
    finally:
        logging.info("Script execution finished.")

# Entry point of the script
if __name__ == "__main__":
    main()