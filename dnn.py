#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries and loading data

# In[2]:


# Import necessary libraries
import numpy as np  # Import NumPy for handling numerical operations
import pandas as pd  # Import Pandas for data manipulation and analysis
import warnings  # Import Warnings to suppress unnecessary warnings

# Suppress warning messages
warnings.filterwarnings("ignore")

# Import matplotlib for data visualization
import matplotlib.pyplot as plt

# Import CatBoostRegressor for building a regression model

# Import mean_squared_error for evaluating model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import train_test_split for splitting the data into training and testing sets
from sklearn.model_selection import train_test_split


# Import RareLabelEncoder from feature_engine.encoding for encoding categorical features
from feature_engine.encoding import RareLabelEncoder

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Import CountVectorizer from sklearn.feature_extraction.text for text feature extraction
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from keras import layers, optimizers, callbacks, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set Pandas options to display a maximum of 1000 rows
pd.set_option('display.max_rows', 1000)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df2 = pd.read_csv('airbnb-listings.csv', delimiter = ';').drop(['ID', 'Listing Url', 'Scrape ID', 'Last Scraped'], axis=1)
df2.head()


# In[4]:


get_ipython().run_cell_magic('time', '', '# Read the dataset from a CSV file into a Pandas DataFrame\ndf = pd.read_csv(\'airbnb-listings.csv\', delimiter = \';\').drop([\'ID\', \'Listing Url\', \'Scrape ID\', \'Last Scraped\'], axis=1)\nitem0 = df.shape[0]  # Stores the initial number of rows in the DataFrame\ndf = df.drop_duplicates()  # Removes duplicate rows from the DataFrame\nitem1 = df.shape[0]  # Stores the number of rows after removing duplicates\nprint(f"There are {item0-item1} duplicates found in the dataset")  # Prints the number of duplicates that were removed\n\n# Select only record with given price\ndf = df[df[\'Price\']>0]\n\n# Replace some locations with more common values\nreplacement_dict = {\'Αθήνα, Greece\': \'Athens, Greece\',\n                    \'Athina, Greece\': \'Athens, Greece\',\n                    \'Roma, Italy\': \'Rome, Italy\',\n                    \'Venezia, Italy\': \'Venice, Italy\',\n                    \'København, Denmark\': \'Copenhagen, Denmark\',\n                    \'Montréal, Canada\': \'Montreal, Canada\',\n                    \'Ville de Québec, Canada\': \'Québec, Canada\',\n                    \'Genève, Switzerland\': \'Geneva, Switzerland\',\n                    \'Palma, Spain\': \'Palma de Mallorca, Spain\',\n                    \'Wien, Austria\': \'Vienna, Austria\',\n                    \'Greater London, United Kingdom\': \'London, United Kingdom\'\n                   }\ndf[\'Smart Location\'] = df[\'Smart Location\'].replace(replacement_dict).fillna(\'None\').astype(str)\n\n# Show only selected columns\nselected_cols = [\'Price\', \'Smart Location\', \'Room Type\', \'Property Type\', \'Bed Type\', \'Availability 365\', \'Minimum Nights\', \'Number of Reviews\', \'Review Scores Rating\', \'Cancellation Policy\']\ndf = df[selected_cols]\nprint(df.shape)  # Prints the dimensions (rows and columns) of the filtered DataFrame\ndf.sample(5).T  # Displays a random sample of 5 rows transposed for better visibility\n')


# In[5]:


df.info()


# ## Data visualisation

# ## Data transformation

# In[6]:


# Accessing DataFrame columns
# This line of code retrieves the column names from a DataFrame called 'df'.
# It allows you to access and work with the names of the columns in the DataFrame.

df.columns


# In[7]:


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


# In[8]:


df.describe().T


# In[9]:


df.dtypes


# ## Machine learning

# In[10]:


# Initialize data: Here, we set up the variables for our machine learning model.
# - 'y' represents the target variable by extracting the values from the 'main_label' column in the DataFrame.
# - 'X' contains the features; we remove the 'main_label' column from the DataFrame to create it.
# - 'cat_cols' stores the column names of categorical features (columns with 'object' data type).
# - 'cat_cols_idx' is a list that stores the indices of categorical columns in the 'X' DataFrame.

y = df[main_label].values.reshape(-1,)
X = df.drop([main_label], axis=1)
cat_cols = df.select_dtypes(include=['object']).columns
cat_cols_idx = [list(X.columns).index(c) for c in cat_cols]

# Split the data into training and testing sets:
# - 'X_train' and 'y_train' contain the feature and target values for the training set, respectively.
# - 'X_test' and 'y_test' contain the feature and target values for the testing set, respectively.
# - We use 'train_test_split' to split the data, with a fixed random state for reproducibility.
# - We also use stratification to ensure balanced class distribution.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=df[['Smart Location']])


# Print the shapes of the resulting datasets to check their dimensions.
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### DNN

# In[11]:


early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',             # Which metric to monitor
    min_delta=0.001,                # Minimum change to qualify as improvement
    patience=15,                    # How many epochs with no improvement to stop training
    verbose=1,                      # Logging level
    restore_best_weights=True       # Restore model weights from the epoch with the best value of the monitored metric.
)

model_checkpoint = ModelCheckpoint('C:/Users/maxim/PycharmProjects/Master_Diploma/models/DNN.h5', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)

callbacks_list = [early_stopping, model_checkpoint, reduce_lr]


# In[12]:


# Preprocessing: Standardizing numerical data and one-hot encoding categorical data
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))])

# We create the column transformer that will allow us to preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)])

# We will preprocess X_train and X_test
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# In[13]:


get_ipython().run_cell_magic('time', '', "# Build the DNN model\nmodel = keras.Sequential([\n    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train_preprocessed.shape[1],)),\n    layers.Dropout(0.5),  # Adding dropout after the first dense layer with a dropout rate of 50%\n    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),\n    layers.Dropout(0.3),  # Adding another dropout layer with a different rate\n    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),\n    layers.Dropout(0.2),  # You can adjust the dropout rates as needed\n    layers.Dense(1)   # For regression, no activation function is used in the output layer\n])\n\n# Set up the learning rate schedule\nlr_schedule = optimizers.schedules.ExponentialDecay(\n    initial_learning_rate=1e-3,  # Starting learning rate\n    decay_steps=10000,  # After this many steps, the learning rate will be decayed\n    decay_rate=0.9)  # The rate at which learning rate will decay\n\n# Use the learning rate schedule with an optimizer (Adam in this case)\noptimizer = optimizers.Adam(learning_rate=lr_schedule)\n\n# Compile the model\nmodel.compile(optimizer=optimizer,\n              loss='mean_squared_error',  # Using Mean Squared Error for a regression problem\n              metrics=['mean_squared_error', 'mean_absolute_error'])\n\n# Train the model\nmodel.fit(\n    X_train_preprocessed, y_train,\n    epochs=200,\n    batch_size=32,\n    verbose=1,\n    validation_data=(X_test_preprocessed, y_test),  # Pass the preprocessed validation data here\n    callbacks= callbacks_list\n)\n")


# In[14]:


# model.save('C:/Users/maxim/PycharmProjects/Master_Diploma/models/DNN.h5')


# In[15]:


# Make predictions
y_train_pred = model.predict(X_train_preprocessed).flatten()
y_test_pred = model.predict(X_test_preprocessed).flatten()

# Calculate metrics
mse_train = mean_squared_error(y_train, y_train_pred, squared=False)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the metrics
print(f"Training Data Metrics:\nMSE: {round(mse_train,2)}\nMAE: {round(mae_train,2)}\nR2 Score: {round(r2_train,2)}")
print(f"\nTesting Data Metrics:\nMSE: {round(mse_test,2)}\nMAE: {round(mae_test,2)}\nR2 Score: {round(r2_test,2)}")


# In[16]:


from IPython.display import display

plt.plot(y_test[:100])
plt.plot(y_test_pred[:100], label='y_test_pred')
plt.show()

# display(y_test[:20])
# display(y_test_pred[:20])


# In[17]:


plt.figure(figsize=(10, 6))  # Set a larger figure size for better readability
plt.plot(y_test[:100], label='Actual', color='blue')  # Add label and color for actual
plt.plot(y_test_pred[:100], label='Predicted', color='orange')  # Already labeled predicted
plt.title('Actual vs. Predicted Rental Prices for the First 100 Test Data Points')  # Add a title
plt.xlabel('Data Point Index')  # Label x-axis
plt.ylabel('Price')  # Label y-axis
plt.legend()  # Display legend
plt.grid(True)  # Add gridlines
plt.show()

