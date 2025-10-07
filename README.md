# Matrix AI CLI

A powerful command-line interface for generating and managing synthetic datasets using advanced AI models.

## 📖 Overview

Matrix AI CLI provides a simple yet powerful way to create high-fidelity synthetic data that mirrors the statistical properties of your original data. It's designed for developers, data scientists, and ML engineers who need realistic, privacy-safe data for testing, development, and model training.

## ✨ Key Features

* **High-Fidelity Synthetic Data**: Leverages the Synthetic Data Vault (SDV) libraries to create statistically accurate data.
* **AI-Powered Schemas**: Uses AI to automatically infer schemas and relationships.
* **Simple Command-Line Interface**: Easy to integrate into your existing scripts and data workflows.
* **Secure Authentication**: Integrates with Google Cloud for secure access to your cloud resources.

## ⚙️ Installation

To get started, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/pavanthalla-a11y/matrix-ai-cli.git

# Navigate to the project directory
cd matrix-ai-cli

# Install dependencies
pip install -r requirements.txt

# Install the CLI tool
pip install .

Example : 

#Sample command

matrix-ai --description "A multi table database with tables customers and transactions" --records 100 --output ./bank_data/
(The csv files will be stored in the same directory folder named /bank_data)

Output in Terminal :

Matrix AI - Synthetic Data Generator

⠋ Calling AI to design schema...WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1759803798.733299    1896 alts_credentials.cc:93] ALTS creds ignored. Not running on GCP and untrusted ALTS is not enabled.
Schema design complete. Please review.

Generated Schema:
{
  "tables": {
    "customers": {
      "columns": {
        "customer_id": {
          "sdtype": "id"
        },
        "first_name": {
          "sdtype": "categorical"
        },
        "last_name": {
          "sdtype": "categorical"
        },
        "email": {
          "sdtype": "categorical"
        },
        "phone_number": {
          "sdtype": "categorical"
        },
        "address": {
          "sdtype": "categorical"
        },
        "city": {
          "sdtype": "categorical"
        },
        "state": {
          "sdtype": "categorical"
        },
        "zip_code": {
          "sdtype": "categorical"
        },
        "join_date": {
          "sdtype": "datetime",
          "datetime_format": "%Y-%m-%d %H:%M:%S"
        }
      },
      "primary_key": "customer_id"
    },
    "transactions": {
      "columns": {
        "transaction_id": {
          "sdtype": "id"
        },
        "customer_id": {
          "sdtype": "id"
        },
        "transaction_date": {
          "sdtype": "datetime",
          "datetime_format": "%Y-%m-%d %H:%M:%S"
        },
        "amount": {
          "sdtype": "numerical"
        },
        "payment_method": {
          "sdtype": "categorical"
        },
        "product_category": {
          "sdtype": "categorical"
        }
      },
      "primary_key": "transaction_id"
    }
  },
  "relationships": [
    {
      "parent_table_name": "customers",
      "child_table_name": "transactions",
      "parent_primary_key": "customer_id",
      "child_foreign_key": "customer_id"
    }
  ]
}

Seed Data Preview:
                                                                           customers                                                                           
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓ 
┃ customer_id ┃ first_name  ┃ last_name ┃ email                     ┃ phone_number ┃ address          ┃ city         ┃ state ┃ zip_code ┃ join_date           ┃ 
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩ 
│ CUST-1001   │ John        │ Smith     │ john.smith@email.com      │ 212-555-0101 │ 123 Maple St     │ New York     │ NY    │ 10001    │ 2022-01-15 10:30:00 │ 
│ CUST-1002   │ Jane        │ Doe       │ jane.d@webmail.com        │ 310-555-0102 │ 456 Oak Ave      │ Los Angeles  │ CA    │ 90001    │ 2022-03-22 14:00:00 │ 
│ CUST-1003   │ Michael     │ Johnson   │ michael.j@corp.com        │ 312-555-0103 │ 789 Pine Ln      │ Chicago      │ IL    │ 60601    │ 2022-05-10 09:15:00 │ 
│ CUST-1004   │ Emily       │ Williams  │ emily.williams@email.com  │ 713-555-0104 │ 101 Birch Rd     │ Houston      │ TX    │ 77001    │ 2022-07-01 11:45:00 │ 
│ CUST-1005   │ David       │ Brown     │ d.brown@inbox.com         │ 215-555-0105 │ 212 Cedar Blvd   │ Philadelphia │ PA    │ 19101    │ 2022-08-19 18:00:00 │ 
│ CUST-1006   │ Sarah       │ Miller    │ sarah.miller@webmail.com  │ 602-555-0106 │ 333 Elm St       │ Phoenix      │ AZ    │ 85001    │ 2023-01-05 12:00:00 │ 
│ CUST-1007   │ Christopher │ Davis     │ chris.davis@corp.com      │ 210-555-0107 │ 444 Spruce Ct    │ San Antonio  │ TX    │ 78201    │ 2023-02-14 20:30:00 │ 
│ CUST-1008   │ Jessica     │ Garcia    │ jess.garcia@email.com     │ 858-555-0108 │ 555 Willow Way   │ San Diego    │ CA    │ 92101    │ 2023-04-30 16:20:00 │ 
│ CUST-1009   │ Daniel      │ Martinez  │ daniel.martinez@inbox.com │ 214-555-0109 │ 666 Redwood Dr   │ Dallas       │ TX    │ 75201    │ 2023-06-11 08:00:00 │ 
│ CUST-1010   │ Amanda      │ Rodriguez │ amanda.r@webmail.com      │ 408-555-0110 │ 777 Sequoia Path │ San Jose     │ CA    │ 95101    │ 2023-08-01 13:10:00 │ 
└─────────────┴─────────────┴───────────┴───────────────────────────┴──────────────┴──────────────────┴──────────────┴───────┴──────────┴─────────────────────┘ 
                                           transactions                                            
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ transaction_id ┃ customer_id ┃ transaction_date    ┃ amount ┃ payment_method ┃ product_category ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ TRX-5001       │ CUST-1001   │ 2022-02-01 11:00:00 │ 125.5  │ Credit Card    │ Electronics      │
│ TRX-5002       │ CUST-1002   │ 2022-03-25 15:30:00 │ 45.2   │ Debit Card     │ Groceries        │
│ TRX-5003       │ CUST-1001   │ 2022-04-10 19:05:00 │ 78.99  │ Credit Card    │ Clothing         │
│ TRX-5004       │ CUST-1003   │ 2022-06-15 09:45:00 │ 299.99 │ PayPal         │ Home Goods       │
│ TRX-5005       │ CUST-1004   │ 2022-07-02 12:15:00 │ 19.95  │ Debit Card     │ Books            │
│ TRX-5006       │ CUST-1005   │ 2022-09-01 14:20:00 │ 550.0  │ Bank Transfer  │ Electronics      │
│ TRX-5007       │ CUST-1002   │ 2022-09-18 10:00:00 │ 88.4   │ Debit Card     │ Groceries        │
│ TRX-5008       │ CUST-1006   │ 2023-01-20 18:50:00 │ 150.75 │ Credit Card    │ Automotive       │
│ TRX-5009       │ CUST-1007   │ 2023-03-01 21:00:00 │ 65.0   │ PayPal         │ Clothing         │
│ TRX-5010       │ CUST-1008   │ 2023-05-05 17:30:00 │ 420.0  │ Credit Card    │ Home Goods       │
│ TRX-5011       │ CUST-1009   │ 2023-06-12 09:30:00 │ 25.5   │ Debit Card     │ Books            │
│ TRX-5012       │ CUST-1010   │ 2023-08-10 11:00:00 │ 899.99 │ Bank Transfer  │ Electronics      │
│ TRX-5013       │ CUST-1001   │ 2023-09-05 14:15:00 │ 55.0   │ Credit Card    │ Groceries        │
│ TRX-5014       │ CUST-1003   │ 2023-10-20 10:20:00 │ 32.8   │ PayPal         │ Books            │
│ TRX-5015       │ CUST-1007   │ 2023-11-11 11:11:00 │ 112.3  │ Credit Card    │ Automotive       │
└────────────────┴─────────────┴─────────────────────┴────────┴────────────────┴──────────────────┘

Do you want to [A]pprove, [R]efine, or [Q]uit? [A]: A

Schema approved. Starting data synthesis...

AI-based constraint validation ━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   3% -:--:--E0000 00:00:1759803948.262037    1896 alts_credentials.cc:93] ALTS creds ignored. Not running on GCP and untrusted ALTS is not enabled.
Success! All foreign keys have referential integrity.

  Table Name  # Rows (Original)  # Invalid Rows  # Rows (New)
   customers                 10               0            10
transactions                 15               0            15
Preprocess Tables:   0%|                                                                                                                 | 0/2 [00:00<?, ?it/s] 
Preprocess Tables:  50%|####################################################5                                                    | 1/2 [00:00<00:00,  6.68it/s] 
Preprocess Tables: 100%|#########################################################################################################| 2/2 [00:00<00:00,  9.19it/s] 


Learning relationships:
(1/1) Tables 'customers' and 'transactions' ('customer_id'):   0%|                                                                      | 0/10 [00:00<?, ?it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  10%|######2                                                       | 1/10 [00:00<00:02,  3.16it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  20%|############4                                                 | 2/10 [00:00<00:01,  4.19it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  30%|##################5                                           | 3/10 [00:00<00:01,  4.73it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  70%|###########################################4                  | 7/10 [00:00<00:00, 11.30it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'): 100%|#############################################################| 10/10 [00:00<00:00, 11.51it/s] 
Learning relationships:
(1/1) Tables 'customers' and 'transactions' ('customer_id'):   0%|                                                                      | 0/10 [00:00<?, ?it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  10%|######2                                                       | 1/10 [00:00<00:02,  3.16it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  20%|############4                                                 | 2/10 [00:00<00:01,  4.19it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  30%|##################5                                           | 3/10 [00:00<00:01,  4.73it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  70%|###########################################4                  | 7/10 [00:00<00:00, 11.30it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'): 100%|#############################################################| 10/10 [00:00<00:00, 11.51it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):   0%|                                                                      | 0/10 [00:00<?, ?it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  10%|######2                                                       | 1/10 [00:00<00:02,  3.16it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  20%|############4                                                 | 2/10 [00:00<00:01,  4.19it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  30%|##################5                                           | 3/10 [00:00<00:01,  4.73it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  70%|###########################################4                  | 7/10 [00:00<00:00, 11.30it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'): 100%|#############################################################| 10/10 [00:00<00:00, 11.51it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  20%|############4                                                 | 2/10 [00:00<00:01,  4.19it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  30%|##################5                                           | 3/10 [00:00<00:01,  4.73it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  70%|###########################################4                  | 7/10 [00:00<00:00, 11.30it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'): 100%|#############################################################| 10/10 [00:00<00:00, 11.51it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  30%|##################5                                           | 3/10 [00:00<00:01,  4.73it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  70%|###########################################4                  | 7/10 [00:00<00:00, 11.30it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'): 100%|#############################################################| 10/10 [00:00<00:00, 11.51it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'):  70%|###########################################4                  | 7/10 [00:00<00:00, 11.30it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'): 100%|#############################################################| 10/10 [00:00<00:00, 11.51it/s] 


Modeling Tables:   0%|                                                                                                                   | 0/2 [00:00<?, ?it/s] 
(1/1) Tables 'customers' and 'transactions' ('customer_id'): 100%|#############################################################| 10/10 [00:00<00:00, 11.51it/s] 


Modeling Tables:   0%|                                                                                                                   | 0/2 [00:00<?, ?it/s] 
Modeling Tables:  50%|#####################################################5                                                     | 1/2 [00:00<00:00,  1.40it/s] 
Modeling Tables: 100%|###########################################################################################################| 2/2 [00:01<00:00,  2.12it/s] 
Modeling Tables: 100%|###########################################################################################################| 2/2 [00:01<00:00,  1.97it/s] 


Modeling Tables:   0%|                                                                                                                   | 0/2 [00:00<?, ?it/s] 
Modeling Tables:  50%|#####################################################5                                                     | 1/2 [00:00<00:00,  1.40it/s] 
Modeling Tables: 100%|###########################################################################################################| 2/2 [00:01<00:00,  2.12it/s] 
Modeling Tables: 100%|###########################################################################################################| 2/2 [00:01<00:00,  1.97it/s] 
Modeling Tables:  50%|#####################################################5                                                     | 1/2 [00:00<00:00,  1.40it/s] 
Modeling Tables: 100%|###########################################################################################################| 2/2 [00:01<00:00,  2.12it/s] 
Modeling Tables: 100%|###########################################################################################################| 2/2 [00:01<00:00,  1.97it/s] 




Enforcing precise row-level constraints with AI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━  88% 0:00:04E0000 00:00:1759803976.649586    1896 alts_credentials.cc:9https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk.
  warning_logs.show_deprecation_warning()
Enforcing precise row-level constraints with AI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━  88% 0:00:04E0000 00:00:1759803976.649586    1896 alts_credentials.cc:9  warning_logs.show_deprecation_warning()
Enforcing precise row-level constraints with AI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━  88% 0:00:04E0000 00:00:1759803976.649586    1896 alts_credentials.cc:9Enforcing precise row-level constraints with AI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━  88% 0:00:04E0000 00:00:1759803976.649586    1896 alts_credentials.cc:93] ALTS creds ignored. Not running on GCP and untrusted ALTS is not enabled.
Synthesis complete. ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

Saving data to ./bank_data/...
- Saved ./bank_data/customers.csv
- Saved ./bank_data/transactions.csv

Data generation complete.


