## Fake News Detection Using LSTM Model

This is a project that demonstrates how to build an LSTM model to detect fake news. The dataset used is the Fake News Detection dataset from Kaggle.

### Prerequisites

This project requires Python 3.x and the following Python libraries installed:

- Pandas
- TensorFlow
- NLTK
- Scikit-learn

### Dataset

The dataset used in this project is the Fake News Detection dataset from Kaggle. The dataset contains two columns, "title" and "text", and a label column "label" that indicates whether the news is real (0) or fake (1). The dataset has a total of 20800 rows.

### Data Preprocessing

After importing the necessary libraries and loading the dataset, we first drop all the rows that have missing values. We then drop the "id" and "author" columns as they are not useful for our analysis. Next, we perform data preprocessing by removing all non-alphabetic characters, converting all text to lowercase, tokenizing the text, removing stop words and performing stemming. Finally, we combine the "title" and "text" columns and create a new column called "corpus".

### One-Hot Representation

We then perform one-hot encoding on the corpus using a vocabulary size of 10000.

### Embedding Representation

Next, we perform padding on the one-hot encoded corpus to ensure that all the sequences have the same length. We then create an embedding layer with 100 embedding vector features, add a dropout layer with a rate of 0.3 to prevent overfitting, add an LSTM layer with 100 units, add another dropout layer with a rate of 0.3, and add a dense layer with sigmoid activation to obtain a binary classification output. We compile the model with binary cross-entropy loss and the Adam optimizer.

### Model Training

We then split the dataset into training and testing sets, with a test size of 33%. We fit the model to the training set for 10 epochs and a batch size of 64. We then use the trained model to predict the labels of the test set and evaluate the accuracy using confusion matrix and accuracy score.

### Model Saving

Finally, we save the trained model in JSON format and the weights in HDF5 format for later use.

### Conclusion

This project demonstrates how to build an LSTM model to detect fake news using the Fake News Detection dataset. By combining the "title" and "text" columns, we can achieve an accuracy score of 90% using the LSTM model.
