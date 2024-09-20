# ML-IDS: Machine Learning-based Intrusion Detection System

## Overview

ML-IDS is a machine learning-based intrusion detection system that utilizes the CICIDS2017 dataset for training and evaluation. This project implements three different machine learning models: Random Forest, Isolation Forest, and Neural Network, to detect various types of network intrusions and cyber attacks.

## Dataset

This project uses the CICIDS2017 dataset, which is a comprehensive intrusion detection dataset created by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB). The dataset contains benign and the most up-to-date common attacks, which resembles the true real-world data (PCAPs).

For more information about the dataset, visit: [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

## Features

- Data preprocessing and feature engineering
- Implementation of three machine learning models:
  - Random Forest
  - Isolation Forest
  - Neural Network
- Model evaluation and performance comparison
- Visualization of results (confusion matrices, feature importance)

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- imbalanced-learn

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/tks98/ml-ids.git
   cd ml-ids
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Download the [CICIDS2017 dataset](http://205.174.165.80/CICDataset/CIC-IDS-2017/) and place the MachineLearningCVE CSV files in the `data/MachineLearningCVE` directory.

2. Run the main script:
   ```
   python main.py
   ```

3. The script will process the data, train the models, and generate results in the `results` directory.

## Project Structure

- `main.py`: The main script that orchestrates the entire process
- `data/`: Directory to store the CICIDS2017 dataset
- `model_cache/`: Directory to store cached models for faster subsequent runs
- `results/`: Directory to store output results and visualizations

## Results

The script generates several output files in the `results` directory:

- Confusion matrices for each model
- Feature importance plot for the Random Forest model
- Overall performance comparison of all models
- CSV file containing anomalies detected by the Isolation Forest model

## Citation

If you use this project or the CICIDS2017 dataset in your research, please cite the following paper:

```
@inproceedings{sharafaldin2018toward,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Habibi Lashkari, Arash and Ghorbani, Ali A},
  booktitle={4th International Conference on Information Systems Security and Privacy (ICISSP)},
  pages={268--282},
  year={2018},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB) for providing the CICIDS2017 dataset.

