# 📝 Next Word Predictor using LSTM

A beautiful **Streamlit Web Application** that predicts the next word in a sentence using a **Long Short-Term Memory (LSTM)** neural network trained on Shakespeare's **Hamlet** dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Features

- 🔮 Predicts the next word from user input
- 🧠 LSTM Deep Learning Model
- 🎨 Beautiful Streamlit UI
- 📊 Confidence Score
- 🏆 Top 5 Predicted Words
- 📜 Prediction History
- ⚡ Fast Model Loading
- 💻 Responsive Layout
- 🌙 Modern Dark Theme

---

<!-- # 📷 Application Preview

> Add your screenshot here

```
project/
│
├── screenshots/
│      app.png
│
└── README.md
```

Then display it using

```md
![App Screenshot](screenshots/app.png)
```

--- -->

# 📂 Project Structure

```text
NextWordPredictor/
│
├── app.py
├── next_word_lstm.h5
├── tokenizer.pickle
├── requirements.txt
├── README.md
└── screenshots/
```

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pickle

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/Aman07-CSE/NextWordPredictor.git

cd NextWordPredictor
```

---

## Create Virtual Environment

### Using Conda

```bash
conda create -n nextword python=3.10

conda activate nextword
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 🧠 Model Information

| Property | Value |
|----------|-------|
| Model | LSTM |
| Framework | TensorFlow |
| Dataset | Shakespeare Hamlet |
| Task | Next Word Prediction |
| Output | Most Probable Next Word |

---

# 💡 Example

### Input

```
To be or not to be
```

### Output

```
that
```

Confidence

```
94.73%
```

---

# 📊 Application Workflow

```
User Input
      │
      ▼
Text Tokenization
      │
      ▼
Padding Sequence
      │
      ▼
LSTM Model
      │
      ▼
Prediction Probability
      │
      ▼
Top 5 Words
      │
      ▼
Display Result
```

---

# ✨ Features Included

- Beautiful Dashboard
- Hero Section
- Responsive Layout
- Sidebar Information
- Prediction History
- Confidence Progress Bar
- Top 5 Predictions
- Sample Prompts
- Error Handling

---

# 📈 Future Improvements

- Sentence Auto Completion
- Beam Search Prediction
- GPT Integration
- Attention Mechanism
- Transformer Model
- Voice Input
- Theme Switcher
- Export Predictions
- Docker Deployment
- Hugging Face Deployment

---


# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create your feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Aman Singh
## ⭐ If you found this project useful, don't forget to star the repository.