# Smart Agricultural Production Optimization Engine (OptiCrop) 🌱

OptiCrop is a simple, lightweight, and professional Machine Learning web application that predicts the most suitable crop for cultivation based on soil chemical composition and environmental indicators. 

This application uses a lightweight **Random Forest Classifier** trained on a realistic synthetic agricultural dataset centering around 22 crop categories.

---

## 🚀 Key Features

* **Advanced Crop Recommendation**: Analyze 7 critical parameters: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Soil pH, and Rainfall.
* **Soil Diagnostics Report**: Compare your inputs against the crop's ideal growth ranges and get custom agronomic advice/remedial actions (e.g., lime treatment for acidic soils, nitrogen additions, drainage improvements).
* **Test Presets**: Populate input values instantly with default presets (e.g., Rice, Chickpea, Cotton, Coffee) to immediately see predictions.
* **Modern Interactive Dashboard**: Premium responsive design styled with an agriculture-themed green & white color palette, transitions, subtle animations, and an animated SVG suitability gauge.
* **Ultra-Lightweight**: Entire project is under 1 MB, leaving a zero-footprint backend and ensuring rapid startup.

---

## 🛠️ Tech Stack

* **Backend**: Python, Flask (Web server and API)
* **Machine Learning**: NumPy, Pandas, Scikit-learn (Random Forest Classifier)
* **Frontend**: HTML5 (Semantic elements), CSS3 (Custom design system), JavaScript (ES6, dynamic presets and gauge animations)

---

## 💻 Installation & Running Instructions

Follow these steps to set up and run OptiCrop on your system.

### 1. Extract or Navigate to the Workspace Directory
Open your terminal/command prompt and navigate to the project directory:
```bash
cd "SAPOE 3"
```

### 2. Install Dependencies
Install all required Python libraries using pip:
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Model (Optional)
The Flask server will automatically detect if the dataset and model are missing and run the training script on startup. If you want to train it manually or inspect the training performance, run:
```bash
python train_model.py
```
This generates:
* `crop_recommendation.csv`: A synthetic agricultural dataset containing 2,200 samples.
* `crop_model.pkl`: A lightweight, serialized Random Forest model (~250 KB) with >99% test accuracy.

### 4. Launch the Web Application
Start the Flask development server:
```bash
python app.py
```

### 5. Access the Web App
Open your web browser and navigate to:
```
http://localhost:5000/
```

---

## 📊 Parameters Analyzed

| Metric | Unit | Target Bounds | Primary Agronomic Role |
| :--- | :--- | :--- | :--- |
| **Nitrogen (N)** | mg/kg (ppm) | 0 - 140 | Boosts foliage and leaf surface development |
| **Phosphorus (P)** | mg/kg (ppm) | 5 - 145 | Strengthens root systems and aids flowering |
| **Potassium (K)** | mg/kg (ppm) | 5 - 205 | Protects from pests and regulates water loss |
| **Temperature** | °C | 10.0 - 50.0 | Affects crop enzyme activity and growth rates |
| **Humidity** | % | 15.0 - 100.0 | Controls transpiration rate and foliage health |
| **pH Level** | pH Scale | 3.5 - 10.0 | Governs nutrient solubility and uptake efficiency |
| **Rainfall** | mm | 20.0 - 300.0 | Provides base moisture for soil and roots |

---

## 📂 Project Structure

```text
SAPOE 3/
│
├── app.py                  # Flask main entrypoint (routes, prediction & diagnostics)
├── train_model.py          # Synthetic dataset generator & Random Forest training code
├── requirements.txt        # Project dependencies (Flask, NumPy, Pandas, Scikit-learn)
├── .gitignore              # Config to exclude cache, venvs, and IDE configurations
├── README.md               # User manual and technical overview
│
├── templates/              # HTML layout documents
│   ├── base.html           # Main template containing layout wrapper
│   ├── home.html           # Welcome landing page
│   ├── recommend.html      # Crop recommendation input form
│   ├── result.html         # Prediction & diagnostics card
│   └── about.html          # ML model details and tech stack info
│
└── static/                 # Static styles & scripts
    ├── css/
    │   └── style.css       # Core stylesheet (color variables, responsive grid, animations)
    └── js/
        └── script.js       # Dynamic presets and animated circular gauge logic
```
