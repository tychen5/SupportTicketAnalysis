# Support Ticket Analysis

This Python script, `support_ticket_analysis.py`, is designed to analyze and clean support ticket data by comparing it with product and issue vault data. The script reads data from three CSV files: product data, support ticket data, and issue vault data. It then performs various operations to correct and filter the support ticket data based on the information available in the product and issue vault data.

### Features

1. Reads data from three CSV files: product data, support ticket data, and issue vault data.
2. Prints the columns of each DataFrame for easy reference.
3. Calculates the value counts for the '1st Level Nesting' column in the product data.
4. Extracts unique issue IDs from both support ticket data and issue vault data.
5. Identifies and corrects issue IDs in the support ticket data that need to be updated based on the issue vault data.
6. Filters the support ticket data to only include rows with non-null issue descriptions.
7. Prints the unique issue IDs and value counts for the 'model' column in the support ticket data.
8. Displays the first 60 rows of the product data and the entire issue vault data.

### Requirements

- Python 3.x
- pandas
- numpy

### Usage

1. Update the file paths for `product_name_path`, `support_ticket_path`, and `issue_vault_path` with the correct paths to your CSV files.
2. Run the script using the following command:

```
python support_ticket_analysis.py
```

### Output

The script will print the following information:

- Columns of each DataFrame (product data, support ticket data, and issue vault data)
- Value counts for the '1st Level Nesting' column in the product data
- Unique issue IDs from both support ticket data and issue vault data
- Issue IDs that need to be corrected
- Unique issue IDs and value counts for the 'model' column in the support ticket data
- First 60 rows of the product data
- Entire issue vault data

After running the script, you can further analyze the cleaned and filtered support ticket data as needed.