import pandas as pd
import numpy as np
import os
import logging
import pickle
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import LeakyReLU
import argparse
import datetime

# Set up logging to track the progress and any issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"TensorFlow version: {tf.__version__}")
logging.info(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

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
            
            print(f"Loaded: {filename}")
        
            # Append to the list of DataFrames
            dataframes.append(df)
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

# Function to load and preprocess the data, using caching for efficiency
def load_and_preprocess_data(directory, cache_file='model_cache/full_dataset_cache.pkl'):
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

# Function to perform feature engineering and create new features
def feature_engineering(df):
    """
    Create new features from existing features based on domain knowledge of cybersecurity and network events during attacks.
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Print available columns
    print("Available columns:")
    print(df.columns.tolist())

    # Avoid division by zero
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Helper function to safely access columns
    def safe_column(col_name):
        return df[col_name] if col_name in df.columns else pd.Series(0, index=df.index)
    
    # Feature 1: Packet Rate Ratio (Forward Packets per Second / Backward Packets per Second)
    df['Packet_Rate_Ratio'] = safe_column('Fwd Packets/s') / (safe_column('Bwd Packets/s') + 1e-6)
    
    # Feature 2: Byte Rate Ratio (Total Length of Forward Packets / Total Length of Backward Packets)
    df['Byte_Rate_Ratio'] = safe_column('Total Length of Fwd Packets') / (safe_column('Total Length of Bwd Packets') + 1e-6)
    
    # Feature 3: Packet Size Variability (Max Packet Length - Min Packet Length)
    df['Packet_Size_Variability'] = safe_column('Max Packet Length') - safe_column('Min Packet Length')
    
    # Feature 4: Flow Duration per Packet (Flow Duration / Total Packets)
    df['Flow_Duration_per_Packet'] = safe_column('Flow Duration') / (safe_column('Total Fwd Packets') + safe_column('Total Backward Packets') + 1e-6)
    
    # Feature 5: Total Flag Count (Sum of all flag counts)
    flag_columns = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 
                    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count']
    df['Total_Flag_Count'] = sum(safe_column(col) for col in flag_columns)
    
    # Feature 6: Header Length Ratio (Forward Header Length / Backward Header Length)
    df['Header_Length_Ratio'] = safe_column('Fwd Header Length') / (safe_column('Bwd Header Length') + 1e-6)
    
    # Feature 7: Active Time Ratio (Active Mean / Idle Mean)
    df['Active_Idle_Ratio'] = safe_column('Active Mean') / (safe_column('Idle Mean') + 1e-6)
    
    # Feature 8: Packet Count Difference (Total Forward Packets - Total Backward Packets)
    df['Packet_Count_Diff'] = safe_column('Total Fwd Packets') - safe_column('Total Backward Packets')
    
    # Feature 9: Percentage of Forward Packets (Total Forward Packets / Total Packets)
    df['Fwd_Packet_Percentage'] = safe_column('Total Fwd Packets') / (safe_column('Total Fwd Packets') + safe_column('Total Backward Packets') + 1e-6)
    
    # Feature 10: Flow Bytes per Packet ((Total Length of Fwd Packets + Total Length of Bwd Packets) / Total Packets)
    df['Flow_Bytes_per_Packet'] = (safe_column('Total Length of Fwd Packets') + safe_column('Total Length of Bwd Packets')) / (safe_column('Total Fwd Packets') + safe_column('Total Backward Packets') + 1e-6)
    
    # Return the dataframe with new features
    return df

# Function to prepare features (X) and target (y) for machine learning
def prepare_features_and_target(df, sample_size=1.0, min_samples=10, use_feature_engineering=True):
    # Sample the data based on the specified percentage
    df_sampled = df.sample(frac=sample_size, random_state=42)
    
    # Print column names before feature engineering
    print("Columns before feature engineering:")
    print(df_sampled.columns.tolist())
    
    # Strip whitespace from column names
    df_sampled.columns = df_sampled.columns.str.strip()
    
    # Identify the columns for the label and source file
    label_col = 'Label'
    source_file_col = 'source_file'
    
    print(f"Label column: '{label_col}'")
    print(f"Source file column: '{source_file_col}'")
    
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
    
    # Perform feature engineering to create new features based on domain knowledge
    if use_feature_engineering:
        logging.info("Performing feature engineering...")
        df_sampled = feature_engineering(df_sampled)
    else:
        logging.info("Skipping feature engineering...")
    
    # Remove any duplicate columns
    df_sampled = df_sampled.loc[:, ~df_sampled.columns.duplicated()]
    
    # Print column names after feature engineering
    print("Columns after feature engineering:")
    print(df_sampled.columns.tolist())
    
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
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y, y_encoded, label_encoder

# Function to train and evaluate the Random Forest model
def train_and_evaluate_model(X, y, use_cache=False, model_cache_file='model_cache/trained_model_cache.joblib'):
    logging.info("Starting Random Forest model training and evaluation process...")
    
    if use_cache and os.path.exists(model_cache_file):
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
            ('scaler', StandardScaler()),
            ('smote', SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
        ])
        
        # Train the model
        logging.info("Training Random Forest model...")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Cache the trained model
        logging.info("Saving Random Forest model to cache...")
        joblib.dump(pipeline, model_cache_file)
        
        # Calculate class distribution after SMOTE
        X_resampled, y_resampled = pipeline.named_steps['smote'].fit_resample(
            pipeline.named_steps['scaler'].transform(X_train),
            y_train
        )
        class_counts_after = Counter(y_resampled)
        logging.info(f"Class distribution after SMOTE: {class_counts_after}")
    
    # Make predictions on the test set
    logging.info("Making predictions on test set...")
    y_pred = pipeline.predict(X_test)
    
    # Generate a classification report
    logging.info("Calculating classification report for Random Forest...")
    report = classification_report(y_test, y_pred, zero_division=1)
    print("\nRandom Forest Classification Report:")
    print(report)
    
    logging.info("Random Forest model evaluation completed.")
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    
    return pipeline, X_test, y_test, y_pred, [accuracy, precision, recall, f1]

# Function to train and evaluate the Isolation Forest model
def train_and_evaluate_isolation_forest(X, y, use_cache=False, model_cache_file='model_cache/isolation_forest_model_cache.joblib'):
    logging.info("Starting Isolation Forest training and evaluation process...")
    
    # Map labels to 'BENIGN' and 'ATTACK'
    y_binary = np.where(y == 'BENIGN', 'BENIGN', 'ATTACK')
    
    if use_cache and os.path.exists(model_cache_file):
        logging.info("Loading Isolation Forest model from cache...")
        isolation_forest = joblib.load(model_cache_file)
        
        # Split data into training and testing sets
        _, X_test, _, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
    else:
        # Split data into training and testing sets
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
        
        # From the training data, select only BENIGN samples to train the Isolation Forest
        X_train = X_train_full[y_train_full == 'BENIGN']
        logging.info(f"Training Isolation Forest on normal data. Training samples: {X_train.shape[0]}")
        
        # Initialize the Isolation Forest model
        isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
        
        # Train the model
        logging.info("Training Isolation Forest...")
        start_time = time.time()
        isolation_forest.fit(X_train)
        end_time = time.time()
        logging.info(f"Isolation Forest training completed in {end_time - start_time:.2f} seconds")
        
        # Save the model to cache
        logging.info("Saving Isolation Forest model to cache...")
        joblib.dump(isolation_forest, model_cache_file)
            
    # Use the trained model to predict anomalies on the test set
    logging.info("Predicting anomalies on test set...")
    y_pred_scores = isolation_forest.decision_function(X_test)
    y_pred = isolation_forest.predict(X_test)
    
    # Map the Isolation Forest output to labels
    # Isolation Forest outputs 1 for normal, -1 for anomalies
    # Let's map 1 to 'BENIGN', -1 to 'ATTACK'
    y_pred_mapped = np.where(y_pred == 1, 'BENIGN', 'ATTACK')
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_mapped,
        'Anomaly_Score': y_pred_scores
    })
    
    # Sort by anomaly score (ascending) to get the most anomalous instances first
    results_df = results_df.sort_values('Anomaly_Score')
    
    # Save the anomalies to a CSV file
    anomalies_file = 'results/isolation_forest_anomalies.csv'
    results_df[results_df['Predicted'] == 'ATTACK'].to_csv(anomalies_file, index=False)
    logging.info(f"Anomalies saved to {anomalies_file}")
    
    # Generate a classification report
    logging.info("Calculating classification report for Isolation Forest...")
    report = classification_report(y_test, y_pred_mapped, zero_division=1)
    print("\nIsolation Forest Classification Report:")
    print(report)
    
    logging.info("Isolation Forest evaluation completed.")
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred_mapped)
    precision = precision_score(y_test, y_pred_mapped, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred_mapped, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred_mapped, average='weighted', zero_division=1)
    
    return isolation_forest, X_test, y_test, y_pred_mapped, [accuracy, precision, recall, f1]

# Function to train and evaluate the Neural Network model
def train_and_evaluate_neural_network(X, y_encoded, label_encoder, use_cache=False, model_cache_file='model_cache/neural_network_model_cache.h5'):
    logging.info("Starting Neural Network training and evaluation process...")
    
    if use_cache and os.path.exists(model_cache_file):
        logging.info("Loading Neural Network model from cache...")
        model = load_model(model_cache_file)
        
        # Split data into training and testing sets
        _, X_test, _, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    else:
        # Split data into training and testing sets
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Apply SMOTE to balance the classes
        smote = SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)
        logging.info(f"Class distribution after SMOTE: {Counter(y_train_resampled)}")
        
        # Recalculate class weights after SMOTE
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
        class_weights_dict = dict(zip(np.unique(y_train_resampled), class_weights))
        
        # Build the neural network model
        model = Sequential()
        model.add(Dense(256, input_dim=X_train_resampled.shape[1]))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))
        
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))
        
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))
        
        model.add(Dense(len(label_encoder.classes_), activation='softmax'))
        
        # Compile the model
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train the model
        logging.info("Training Neural Network model...")
        start_time = time.time()
        model.fit(X_train_resampled, y_train_resampled, validation_split=0.2, epochs=50, batch_size=128, class_weight=class_weights_dict, callbacks=[early_stopping])
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Save the model to cache
        logging.info("Saving Neural Network model to cache...")
        model.save(model_cache_file)
        
        # Save the scaler
        with open('model_cache/scaler_nn.pkl', 'wb') as f:
            pickle.dump(scaler, f)
                
    # Evaluate the model on the test set
    logging.info("Evaluating Neural Network model on test set...")
    # Normalize features
    if 'scaler' not in locals():
        # Load scaler from cache or recompute
        with open('model_cache/scaler_nn.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test = scaler.transform(X_test)
    
    y_pred_prob = model.predict(X_test)
    y_pred_encoded = np.argmax(y_pred_prob, axis=1)
    
    # Map labels back to original labels
    y_test = label_encoder.inverse_transform(y_test_encoded)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Generate a classification report
    logging.info("Calculating classification report for Neural Network...")
    report = classification_report(y_test, y_pred, zero_division=1)
    print("\nNeural Network Classification Report:")
    print(report)
    
    logging.info("Neural Network model evaluation completed.")
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    
    return model, X_test, y_test, y_pred, [accuracy, precision, recall, f1]

# Function to plot a confusion matrix of model predictions
def plot_confusion_matrix(y_test, y_pred, top_n=10, title='Confusion Matrix', filename='results/confusion_matrix.png'):
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
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()
    logging.info(f"{title} saved as '{filename}'")

# Function to plot feature importance from the Random Forest model
def plot_feature_importance(pipeline, X, run_dir):
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
    plt.savefig(os.path.join(run_dir, 'feature_importance.png'))
    plt.close()
    
    logging.info(f"Feature importance plot saved in {run_dir}")
    
    # Save feature importances to a CSV file for further analysis
    feature_importances.to_csv(os.path.join(run_dir, 'feature_importances.csv'))
    logging.info(f"Feature importances saved in {run_dir}")

# Function to create an overall performance report
def create_overall_performance_report(rf_results, if_results, nn_results, run_dir):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = ['Random Forest', 'Isolation Forest', 'Neural Network']
    
    data = {
        'Metric': metrics * 3,
        'Score': rf_results + if_results + nn_results,
        'Model': [models[0]] * 4 + [models[1]] * 4 + [models[2]] * 4
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(x='Metric', y='Score', hue='Model', data=df)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title='Model', title_fontsize='12', fontsize='10')
    plt.savefig(os.path.join(run_dir, 'overall_performance_report.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Overall performance report saved in {run_dir}")

# Function to create a unique run directory
def create_run_directory():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('results', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Main function to orchestrate the entire process
def main(args):
    try:
        # Create directories if they don't exist
        os.makedirs('model_cache', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # Create a unique run directory
        run_dir = create_run_directory()
        logging.info(f"Created run directory: {run_dir}")

        # Specify the directory containing the data files
        directory = 'data/MachineLearningCVE'
        # Load and preprocess the data
        full_dataset = load_and_preprocess_data(directory)
        
        # Convert the percentage to a fraction
        sample_size = args.data / 100.0
        logging.info(f"Using {args.data}% of the dataset (sample size: {sample_size})")
        
        # Prepare features and target for machine learning
        X, y, y_encoded, label_encoder = prepare_features_and_target(full_dataset, sample_size=sample_size, use_feature_engineering=args.feature_engineering)
        
        rf_metrics = if_metrics = nn_metrics = None

        if args.classification or not (args.classification or args.nn):
            # Train and evaluate the Random Forest model
            pipeline, X_test_rf, y_test_rf, y_pred_rf, rf_metrics = train_and_evaluate_model(X, y, use_cache=args.use_model_cache)
            
            # Generate and save visualizations for Random Forest
            plot_confusion_matrix(y_test_rf, y_pred_rf, title='Random Forest Confusion Matrix', 
                                  filename=os.path.join(run_dir, 'confusion_matrix_rf.png'))
            logging.info(f"Random Forest confusion matrix saved in {run_dir}")
            
            plot_feature_importance(pipeline, X, run_dir)
            
            # Train and evaluate the Isolation Forest model
            isolation_forest, X_test_if, y_test_if, y_pred_if, if_metrics = train_and_evaluate_isolation_forest(X, y, use_cache=args.use_model_cache)
            
            # Generate and save visualizations for Isolation Forest
            plot_confusion_matrix(y_test_if, y_pred_if, top_n=2, title='Isolation Forest Confusion Matrix', 
                                  filename=os.path.join(run_dir, 'confusion_matrix_if.png'))
            logging.info(f"Isolation Forest confusion matrix saved in {run_dir}")

        if args.nn or not (args.classification or args.nn):
            # Train and evaluate the Neural Network model
            model_nn, X_test_nn, y_test_nn, y_pred_nn, nn_metrics = train_and_evaluate_neural_network(X, y_encoded, label_encoder, use_cache=args.use_model_cache)
            
            # Generate and save visualizations for Neural Network
            plot_confusion_matrix(y_test_nn, y_pred_nn, title='Neural Network Confusion Matrix', 
                                  filename=os.path.join(run_dir, 'confusion_matrix_nn.png'))
            logging.info(f"Neural Network confusion matrix saved in {run_dir}")
        
        # Create and save the overall performance report if we have results from both
        if rf_metrics and nn_metrics:
            create_overall_performance_report(rf_metrics, if_metrics, nn_metrics, run_dir)
        
        logging.info("Process completed!")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    
    finally:
        logging.info("Script execution finished.")

# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine learning models on network traffic data.")
    parser.add_argument("--classification", action="store_true", help="Run Random Forest and Isolation Forest classifiers")
    parser.add_argument("--nn", action="store_true", help="Run Neural Network classifier")
    parser.add_argument("--data", type=float, default=100.0, help="Percentage of dataset to use (default: 100.0)")
    parser.add_argument("--use-model-cache", action="store_true", help="Use cached models if they exist")
    parser.add_argument("--feature-engineering", action="store_true", help="Use feature engineering")
    args = parser.parse_args()

    # If no arguments are provided, run both
    if not (args.classification or args.nn):
        args.classification = args.nn = True

    # Ensure the data percentage is between 0 and 100
    args.data = max(0.0, min(100.0, args.data))

    main(args)