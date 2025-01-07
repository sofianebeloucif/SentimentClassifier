# TextSentimentAnalyzer

## Description

This project aims to create a sentiment analysis model that determines whether a given text is joyfull, sad, angry ect... . The model is trained on the Kaggle dataset Emotions dataset for NLP  and uses TensorFlow/Keras for deep learning. The application is built using Streamlit to provide an interactive web interface where users can input text and receive predictions on the sentiment.

## Features

- **Text Input:** Users can input a sentence or paragraph to analyze its sentiment.
- **Sentiment Prediction:** The model predicts whether the sentiment of the text is joy, anger, sadness, love, fear or surprise.
- **Interactive Interface:** Built with Streamlit to provide a smooth and easy-to-use experience.

## Installation

To install the necessary dependencies, use the following command:

```bash
python.exe -m pip install -r requirements.txt
```

Make sure you have Python 3.8 installed.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/TextSentimentAnalyzer.git
   cd TextSentimentAnalyzer
   ```


2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. After running the above command, the application should open automatically in your web browser. If not, navigate to the URL provided in the terminal.

## Model Details

The sentiment analysis model is based on a **neural network** using **TensorFlow/Keras**. It is trained on the **Kaggle dataset** which contains list of sentences with emotion flag. The model architecture consists of several dense layers with dropout to prevent overfitting.

## File Structure

```
SentimentClassifier/
│
├── src                
│   └── app.py                  # Main Streamlit application file
│   └── preprocess.py           # Preprocessing script for stemming
│   └── Emotion detection.ipynb # Training code
│   └── models/                
│        └── naive_model.h5     # Model architecture
├──      
├── requirements.txt      # List of required Python packages
├── data/                 # Directory containing the dataset
│   └── test.txt          # Kaggle dataset used for training
│   └── result_row.csv    # Data to plot the bar chart 
└── utils/                # Utility scripts (e.g., for preprocessing data)
```

## How to Use the App

To use the **Sentiment Analysis App**, simply visit the following link: [Sentiment Analysis App](https://sentimentclassifier-8appvycedti2bqvjamesxwp.streamlit.app/).

Once on the page:
1. Enter a sentence or text in the input box located in the sidebar.
2. Click anywhere outside the input box to submit your text.
3. The app will analyze the text and display the predicted emotion.

Additionally, the app provides insights into how many words were needed to make accurate predictions on a sample dataset. A bar chart visualizes this information to help you understand the model's behavior.

Feel free to explore and test the app with different inputs!

## Contributing

Feel free to fork the repository, create a branch, and submit pull requests with improvements. Please ensure that you follow the existing coding style and write tests for new features.


## To do

- [ ] Do K cross validation to find the best hyperparameters 
## License

This project is licensed under the MIT License.

