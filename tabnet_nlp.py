#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Import necessary libraries
import numpy as np  # Import NumPy for handling numerical operations
import pandas as pd  # Import Pandas for data manipulation and analysis
import warnings  # Import Warnings to suppress unnecessary warnings

# Suppress warning messages
warnings.filterwarnings("ignore")


# Import matplotlib for data visualization
import matplotlib.pyplot as plt

# Import mean_squared_error for evaluating model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import train_test_split for splitting the data into training and testing sets
from sklearn.model_selection import train_test_split


# Import RareLabelEncoder from feature_engine.encoding for encoding categorical features
from feature_engine.encoding import RareLabelEncoder

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

from pytorch_tabnet.tab_model import TabNetRegressor
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the stopwords from NLTK
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from gensim.models import Word2Vec

# Set Pandas options to display a maximum of 1000 rows
pd.set_option('display.max_rows', 1000)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


df2 = pd.read_csv('airbnb-listings.csv', delimiter = ';').drop(['ID', 'Listing Url', 'Scrape ID', 'Last Scraped'], axis=1)
df2.head()


# ### Text Preprocessing

# In[19]:


get_ipython().run_cell_magic('time', '', '# Read the dataset from a CSV file into a Pandas DataFrame\ndf = pd.read_csv(\'airbnb-listings.csv\', delimiter = \';\').drop([\'ID\', \'Listing Url\', \'Scrape ID\', \'Last Scraped\'], axis=1)\nitem0 = df.shape[0]  # Stores the initial number of rows in the DataFrame\ndf = df.drop_duplicates()  # Removes duplicate rows from the DataFrame\nitem1 = df.shape[0]  # Stores the number of rows after removing duplicates\nprint(f"There are {item0-item1} duplicates found in the dataset")  # Prints the number of duplicates that were removed\n\n# Select only record with given price\ndf = df[df[\'Price\']>0]\n\n# Replace some locations with more common values\nreplacement_dict = {\'Αθήνα, Greece\': \'Athens, Greece\',\n                    \'Athina, Greece\': \'Athens, Greece\',\n                    \'Roma, Italy\': \'Rome, Italy\',\n                    \'Venezia, Italy\': \'Venice, Italy\',\n                    \'København, Denmark\': \'Copenhagen, Denmark\',\n                    \'Montréal, Canada\': \'Montreal, Canada\',\n                    \'Ville de Québec, Canada\': \'Québec, Canada\',\n                    \'Genève, Switzerland\': \'Geneva, Switzerland\',\n                    \'Palma, Spain\': \'Palma de Mallorca, Spain\',\n                    \'Wien, Austria\': \'Vienna, Austria\',\n                    \'Greater London, United Kingdom\': \'London, United Kingdom\'\n                   }\ndf[\'Smart Location\'] = df[\'Smart Location\'].replace(replacement_dict).fillna(\'None\').astype(str)\n')


# In[20]:


get_ipython().run_cell_magic('time', '', "stop_words = set(stopwords.words('english'))\n\ndef preprocess_text(text):\n    # Tokenization\n    tokens = word_tokenize(text)\n    # Convert to lower case and remove punctuation\n    tokens = [word.lower() for word in tokens if word.isalpha()]\n    # Remove stop words\n    tokens = [word for word in tokens if not word in stop_words]\n    return tokens\n\ndf = df.dropna(subset=['Description'])\ndf['processed_description'] = df['Description'].apply(preprocess_text)\n")


# ### Embedding Generation

# In[21]:


# Train Word2Vec on your corpus
model_w2v = Word2Vec(sentences=df['processed_description'].tolist(), vector_size=300, window=5, min_count=1, workers=4)
model_w2v.save("word2vec.model")

def get_vector(words):
    vector = []
    for word in words:
        vector.append(model_w2v.wv[word])
    return np.mean(vector, axis=0) if len(vector) > 0 else np.zeros(300)

df['embedding'] = df['processed_description'].apply(get_vector)


# In[22]:


# Show only selected columns
selected_cols = ['Price', 'Smart Location', 'Room Type', 'Property Type', 'Bed Type', 'Availability 365', 'Minimum Nights', 'Number of Reviews', 'Review Scores Rating', 'Cancellation Policy', 'embedding']
df = df[selected_cols]
print(df.shape)  # Prints the dimensions (rows and columns) of the filtered DataFrame
df.sample(5).T


# ### Add Embeddings to Dataset with PCA

# In[23]:


# PCA reduction
embeddings = np.vstack(df['embedding'].values)
pca = PCA(n_components=50)  # Adjust n_components as needed
reduced_embeddings = pca.fit_transform(embeddings)
for i in range(reduced_embeddings.shape[1]):
    df[f'pca_{i}'] = reduced_embeddings[:, i]

# Dropping original and intermediate columns used for embedding
df = df.drop(columns=['embedding'])  # Adjust column names as needed


# ### Encoding

# In[24]:


# Define the main label column as 'Price'
main_label = 'Price'

# Exclude the 1% of the smallest and 1% of the highest labels in the DataFrame
P = np.percentile(df[main_label], [1, 99])
df = df[(df[main_label] > P[0]) & (df[main_label] < P[1])]

# Function to bin numerical columns into equal quantile-based bins
def bin_column(df, col_name, num_bins=7):
    # Calculate the bin edges to evenly split the numerical column
    bin_edges = pd.qcut(df[col_name], q=num_bins, retbins=True)[1]

    # Define labels for the categorical bins based on bin edges
    bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(num_bins)]

    # Use pd.qcut to create quantile-based bins with an equal number of records in each bin
    df[col_name] = pd.qcut(df[col_name], q=num_bins, labels=False)

    # Update the bin labels to be more descriptive
    df[col_name] = df[col_name].map(lambda x: bin_labels[x])

    # Convert the column to object dtype
    df[col_name] = df[col_name].astype('object')

    return df

# Iterate through DataFrame columns (excluding the main label column)
for col in df.columns:
    if col != main_label:
        try:
            # Bin the column if it's numerical
            df = bin_column(df, col)
            print(f"Binned column {col}")
        except:
            # If not numerical, handle as categorical and apply RareLabelEncoder
            df[col] = df[col].fillna('None').astype(str).apply(lambda x: x.rstrip('.0'))
            encoder = RareLabelEncoder(n_categories=1, max_n_categories=70, replace_with='Other', tol=50/df.shape[0])
            df[col] = encoder.fit_transform(df[[col]])
            print(f"LabelEncoded column {col}")

# Print the shape of the resulting DataFrame
print(df.shape)

# Display a sample of 5 rows from the DataFrame, transposed for easier viewing
df.sample(5).T


# In[30]:


df.describe().T


# ### Data Initialization

# In[31]:


# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Extract target variable and features
y = df['Price'].values
X = df.drop(columns=['Price'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Model Train

# In[32]:


# Initialize TabNet regressor
regressor = TabNetRegressor(optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=2e-2),
                            scheduler_params={"step_size":10, "gamma":0.9},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR,
                            mask_type='entmax'
)

# Train the regressor
regressor.fit(
  X_train.values, y_train.reshape(-1, 1),
  eval_set=[(X_test.values, y_test.reshape(-1, 1))],
  patience=30,  # Stop training if the loss does not decrease for 10 consecutive epochs
  max_epochs=300
)


# In[33]:


regressor.save_model('tabnet_NLP_model')


# In[34]:


# Make predictions
y_train_pred = regressor.predict(X_train.values)
y_test_pred = regressor.predict(X_test.values)

# Calculate performance metrics
mse_train = mean_squared_error(y_train, y_train_pred, squared=False)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the metrics
print(f"Training Data Metrics:\nMSE: {round(mse_train,2)}\nMAE: {round(mae_train,2)}\nR2 Score: {round(r2_train,2)}")
print(f"\nTesting Data Metrics:\nMSE: {round(mse_test,2)}\nMAE: {round(mae_test,2)}\nR2 Score: {round(r2_test,2)}")


# In[35]:


from IPython.display import display

plt.plot(y_test[:100])
plt.plot(y_test_pred[:100], label='y_test_pred')
plt.show()

# display(y_test[:20])
# display(y_test_pred[:20])

