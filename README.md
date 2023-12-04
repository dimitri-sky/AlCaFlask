# Back-End Stock Prediction System

An LSTM stock prediction system that leverages deep learning to predict the stock prices of various companies for the upcoming 3 days. The system is implemented using TensorFlow and Flask, and it fetches real-time data from Yahoo Finance.

<div>
   <h2>Front-End part at: <a href="https://github.com/cod-cs-club/almightycandle">https://www.almightycandle.com/</a> <br>
      Visit Now: <a href="https://www.almightycandle.com/">almightycandle.com</a> <br> Stock Prediction Website. (AWS)</h2> 
</div>


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Predicts the stock prices for the next 3 days for various companies.
- Utilizes LSTM (Long Short-Term Memory) networks for accurate predictions.
- Flask API to provide predictions through an HTTP endpoint.
- Real-time data fetching from Yahoo Finance.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dimitri-sky/AlCaFlask.git
   cd AlCaFlask
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server:

   ```bash
   python main.py
   ```

## Usage

### Predicting Stock Prices

Send a GET request to the `/predict` endpoint with the stock symbol as a parameter.

```bash
curl http://127.0.0.1:5000/predict?symbol=AAPL
```

### Training the Model

Run the `train_model.py` script to train the model on the predefined stocks.

```bash
python train_model.py
```

## Project Structure

- `main.py`: The Flask server that exposes the prediction endpoint.
- `train_model.py`: Script to train the LSTM model on stock data.
- `saved_modelv4.h5`: Pre-trained LSTM model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
