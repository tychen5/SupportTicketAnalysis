IssueCategorization.py

# README.md

## Issue Categorization

This Python code snippet, named `IssueCategorization.py`, is designed to categorize and group issues based on their textual content. The code utilizes various pre-trained transformer models from the Hugging Face library to extract features from the text and generate embeddings. These embeddings are then used to categorize the issues into different groups.

### Key Features

1. Utilizes pre-trained transformer models from Hugging Face library, including:
   - XLM-RoBERTa Base
   - CardiffNLP Twitter XLM-RoBERTa Base
   - Microsoft InfoXLM Base
   - Microsoft XLM-Align Base

2. Customizable preprocessing steps for cleaning and tokenizing text data.

3. Customizable weights for combining embeddings from different models.

4. Efficient memory management using garbage collection and PyTorch's `torch.cuda.empty_cache()`.

### Usage

1. Replace the placeholders in the code with the appropriate paths to your data files and model files.

2. Customize the preprocessing steps and stop words list as needed.

3. Run the `IssueCategorization.py` script to generate embeddings for your text data.

4. Use the generated embeddings for further analysis, such as clustering or classification tasks.

### Dependencies

- Python 3.6 or higher
- PyTorch
- Transformers
- Pandas
- NumPy
- NLTK

### Example

```python
import pandas as pd
from IssueCategorization import prepare_models, extract_features_emb

# Load your data
issue_df = pd.read_pickle('/path/to/your/issue_vault_embeddings.pkl')

# Prepare models, tokenizers, and dataloaders
dataloader_li, device_li, model_li, weight_li = prepare_models("Description_clean", issue_df)

# Extract features and generate embeddings
final_emb = extract_features_emb(dataloader_li, model_li, device_li, weight_li)

# Use the embeddings for further analysis
# ...
```

### Note

Please ensure that you have the necessary dependencies installed and that you have replaced the placeholders in the code with the appropriate paths to your data and model files.