# Model Evaluation Summary

Before building the final system, we evaluated multiple machine-learning models using two different text-representation techniques:

### **1. Word-Level Unigram Features (TF-IDF)**

### **2. Character-Level 7-Gram Features (TF-IDF)**

For each representation, we tested **five classification algorithms**:

* Logistic Regression
* Support Vector Machine (LinearSVC)
* Multinomial Naive Bayes
* Random Forest
* Multi-Layer Perceptron (MLP)

### **Results**

* Overall, **word-level unigram TF-IDF** produced more stable and higher accuracy compared to char 7-gram.
* Among all models, **Logistic Regression consistently achieved the best performance**, giving the most reliable predictions on hate and non-hate text.

Because of this, the final deployed system uses:

✔ **TF-IDF (word unigram)**
✔ **Logistic Regression (max_iter = 10000)**

This combination offered the best accuracy and generalization on our balanced dataset (50,000 hate + 50,000 non-hate examples).


# Hate Speech Detection (TF-IDF + Logistic Regression)

This project is a hate-speech classification system built using:

* TF-IDF (word unigram)
* Logistic Regression
* Streamlit UI
* Joblib for saving and loading the model

The system predicts whether a given text is **Hate Text** or **Not Hate Text**.

---

## Project Links

* **GitHub Repository:**
  [https://github.com/AhmedHussain007/Hate-Speech-Recognition](https://github.com/AhmedHussain007/Hate-Speech-Recognition)

* **Live Deployed App:**
  [https://hate-speech-recognition-y8kwvdguaokd5cdwhx7ir2.streamlit.app/](https://hate-speech-recognition-y8kwvdguaokd5cdwhx7ir2.streamlit.app/)

* **Dataset Used (Mendeley Hate Speech Dataset):**
  [https://data.mendeley.com/datasets/9sxpkmm8xn/1](https://data.mendeley.com/datasets/9sxpkmm8xn/1)

---

## ![image.png](attachment:image.png)

Deployed Model Screenshot
---

## ![image-2.png](attachment:image-2.png)

---

## Example Predictions

### Hate Text

* You should be ashamed of yourself.
* Go away, you are worthless.
* Nobody likes you, just get out from here idiot.

### Not Hate Text

* I hope you have a wonderful day.
* Congratulations on your achievement.
* Thank you for helping me yesterday.

---

## How It Works

1. Dataset cleaned and balanced (50k hate + 50k non-hate).
2. TF-IDF unigram features extracted.
3. Logistic Regression trained and saved as `best_model.pkl`.
4. Streamlit loads the model and predicts labels for user-input text.

---

## Run Locally

```
pip install -r requirements.txt
streamlit run main.py
```
