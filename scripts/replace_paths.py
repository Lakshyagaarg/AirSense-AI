# Text-based replacements to update Colab paths to Windows paths
# This script avoids using `os` or dictionaries — simple string replaces only.

base = r"C:\Users\HP\OneDrive\Desktop\DataScience\AQI Project"

replacements = [
    ("/content/drive/MyDrive/AQI Project/city_day.csv.zip", base + r"\\Data\\city_day.csv.zip"),
    ("/content/drive/MyDrive/AQI Project/station_day.csv.zip", base + r"\\Data\\station_day.csv.zip"),
    ("/content/drive/MyDrive/AQI Project/model_results.csv", base + r"\\Result_csv\\model_results.csv"),
    ("/content/drive/MyDrive/AQI Project/Lasso_model_results.csv", base + r"\\Result_csv\\Lasso_model_results.csv"),
    ("/content/drive/MyDrive/AQI Project/classification_results.csv", base + r"\\Result_csv\\classification_results.csv"),
    ("/content/drive/MyDrive/AQI Project/best_model.pkl", base + r"\\Pickle_Files\\best_model.pkl"),
    ("/content/drive/MyDrive/AQI Project/ridge_model.pkl", base + r"\\Pickle_Files\\ridge_model.pkl"),
    ("/content/drive/MyDrive/AQI Project/scaler.pkl", base + r"\\Pickle_Files\\scaler.pkl"),
]

files = [
    r"c:\\Users\\HP\\OneDrive\\Desktop\\DataScience\\AQI Project\\Notebook.ipynb",
    r"c:\\Users\\HP\\OneDrive\\Desktop\\DataScience\\AQI Project\\app.py",
]

for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        continue

    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)

    with open(fp, "w", encoding="utf-8") as f:
        f.write(text)

print("Replacement complete.")
