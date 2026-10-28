import joblib
model = joblib.load("fake_news_model.joblib")

print("Fake news Detection System")
print("Enter a news article below.")

article = input("\nNews article: ").strip()

if not article:
    print("Please enter some news text.")
else:
    prediction = model.predict([article])[0]

    if prediction == 0:
        print("\n Prediction: Likely Fake")
    else:
        print("\nPrediction: Likely Real")

    print(
        "\nNote: This is a model prediction,"
        "not a verified fact-check."
    )