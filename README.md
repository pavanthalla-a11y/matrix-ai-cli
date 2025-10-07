# Matrix AI CLI

A powerful command-line interface for generating and managing synthetic datasets using advanced AI models.

## 📖 Overview

Matrix AI CLI provides a simple yet powerful way to create high-fidelity synthetic data that mirrors the statistical properties of your original data. It's designed for developers, data scientists, and ML engineers who need realistic, privacy-safe data for testing, development, and model training.

## ✨ Key Features

* **High-Fidelity Synthetic Data**: Leverages the Synthetic Data Vault (SDV) libraries to create statistically accurate data.
* **AI-Powered Schemas**: Uses AI to automatically infer schemas and relationships from your data sources.
* **Simple Command-Line Interface**: Easy to integrate into your existing scripts and data workflows.
* **Secure Authentication**: Integrates with Google Cloud for secure access to your cloud resources.

## ⚙️ Installation

To get started, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone [https://github.com/pavanthalla-a11y/matrix-ai-cli.git](https://github.com/pavanthalla-a11y/matrix-ai-cli.git)

# Navigate to the project directory
cd matrix-ai-cli

# Install dependencies
pip install -r requirements.txt

# Install the CLI tool
pip install .

#Sample command
matrix-ai --description "A multi table database with tables customers and transactions" --records 100 --output ./bank_data/
(The csv files will be stored in the same directory folder named /bank_data)