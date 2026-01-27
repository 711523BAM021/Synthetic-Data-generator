# Synthetic Data Generator

## Project Overview

The Synthetic Data Generator is a web-based AI application that creates realistic artificial datasets from real-world data while preserving privacy. The system automatically analyzes the uploaded dataset and generates high-quality synthetic data suitable for machine learning training, testing, and research.

This project combines data preprocessing, statistical modeling, and deep learning to handle different types of datasets through a simple web interface.

---

## Objective

The main goal of this project is to solve the problem of **data privacy** while still allowing the use of realistic data for:

- Machine learning model training  
- Software testing  
- Academic research  
- Data analysis practice  

Instead of sharing real sensitive data, synthetic data with similar patterns is generated.

---

## Key Features

- Upload any CSV file  
- Automatic data cleaning  
- Detects dataset type automatically  
- Supports:
  - Numerical datasets  
  - Categorical datasets  
  - Mixed datasets  
  - Time-series datasets  
- AI-based synthetic data generation  
- Realism score calculation  
- Distribution comparison graph  
- Synthetic data preview  
- Download cleaned real dataset  
- Download synthetic dataset  

---

## Models Used

### 1. Gaussian Copula Model (Tabular Data)
Used when the dataset is structured without time dependency.  
It learns statistical relationships between columns and generates synthetic data preserving those relationships.

### 2. LSTM Time-Series Model
Used when a date/time column is detected.  
It learns temporal patterns and trends to generate realistic sequential data.

---

## How the System Works

1. User uploads a CSV file through the web interface.  
2. Data cleaning is performed:
   - Remove duplicates  
   - Handle missing values  
   - Standardize formats  
3. System detects dataset type:
   - Time column → Time-series model  
   - No time column → Tabular model  
4. The selected model is trained on the dataset.  
5. Synthetic data is generated.  
6. Real and synthetic distributions are compared.  
7. Realism score is calculated.  
8. Results page displays:
   - Dataset analysis  
   - Model used  
   - Realism score  
   - Limitations  
   - Synthetic preview  
9. User downloads outputs.

---
## Features
- Automatic CSV Analysis
- Smart Data Cleaning
- Automatic Model Selection
- Realism Score
- Graph Comparison
- Data Preview
- Download Options
---

## Folder Structure

<img width="436" height="650" alt="image" src="https://github.com/user-attachments/assets/ae25fbe0-5810-46a3-8cf5-852b0239aff0" />


---

## Technologies Used

- Python  
- Flask  
- Pandas  
- NumPy  
- TensorFlow / Keras  
- SDV (Synthetic Data Vault)  
- Matplotlib  
- HTML & CSS  

---

## Limitations

- Very small datasets reduce realism  
- Time-series generation needs more memory  
- Very high-dimensional data increases processing time  

---

## Author

**Gowthamraj B**  
CSE (Artificial Intelligence and Machine Learning)  
Synthetic Data Generator Project


