import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

# Ensure we can import from train_model if we need to train on the fly
from train_model import generate_and_train, crop_ranges

app = Flask(__name__)
app.secret_key = "opti_crop_secret_key"

MODEL_PATH = "crop_model.pkl"
CSV_PATH = "crop_recommendation.csv"

# Load the Machine Learning model on startup
# If it doesn't exist, train it first
if not os.path.exists(MODEL_PATH) or not os.path.exists(CSV_PATH):
    print("Pre-trained model or dataset not found. Generating dataset and training model...")
    generate_and_train(csv_path=CSV_PATH, model_path=MODEL_PATH)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Metadata for each crop to display rich results
crop_details = {
    'rice': {
        'name': 'Rice',
        'type': 'Cereal / Grain',
        'season': 'Kharif (Monsoon / Wet Season)',
        'duration': '105 - 150 days',
        'description': 'Rice is a staple cereal crop that requires significant water and warm temperatures. It grows best in clayey or loamy soils that can retain moisture.',
        'tips': ['Maintain a shallow layer of water in the field during the early growth stages.', 'Nitrogen management is crucial; apply in split doses.', 'Ensure good field levelling to distribute water uniformly.']
    },
    'maize': {
        'name': 'Maize (Corn)',
        'type': 'Cereal / Grain',
        'season': 'Kharif / Rabi (Year-round)',
        'duration': '90 - 120 days',
        'description': 'Maize is a versatile crop grown for food, fodder, and industrial uses. It prefers well-drained, fertile soils with a neutral pH.',
        'tips': ['Ensure soil has good organic matter and drainage.', 'Critically sensitive to waterlogging, especially at the silking stage.', 'Weed control is highly important in the first 4-6 weeks.']
    },
    'chickpea': {
        'name': 'Chickpea',
        'type': 'Pulse / Legume',
        'season': 'Rabi (Winter / Dry Season)',
        'duration': '90 - 110 days',
        'description': 'Chickpeas are nutritious legumes that enrich the soil by fixing atmospheric nitrogen. They require moderate temperatures and minimal rainfall.',
        'tips': ['Avoid waterlogging at all costs; chickpeas are highly susceptible to root rot.', 'No heavy nitrogen fertilizer is required due to nitrogen-fixing nodules.', 'Protect crop from pod borers during early flowering stages.']
    },
    'kidneybeans': {
        'name': 'Kidney Beans (Rajma)',
        'type': 'Pulse / Legume',
        'season': 'Rabi / Kharif (Cool weather)',
        'duration': '90 - 130 days',
        'description': 'Kidney beans are popular high-protein legumes. They grow best in cool to moderate climates with well-drained, sandy loam soils.',
        'tips': ['Keep soil moist but not soggy.', 'Provide light nitrogen initially before nodules fully establish.', 'Mulch around plants to conserve soil moisture.']
    },
    'pigeonpeas': {
        'name': 'Pigeon Peas (Arhar)',
        'type': 'Pulse / Legume',
        'season': 'Kharif (Monsoon)',
        'duration': '150 - 180 days',
        'description': 'Pigeon peas are deep-rooted shrubs highly resilient to drought. They are often intercropped with cereals like sorghum or pearl millet.',
        'tips': ['Excellent for crop rotation to restore soil fertility.', 'Requires minimal irrigation after initial root establishment.', 'Ensure proper soil drainage to prevent wilt disease.']
    },
    'mothbeans': {
        'name': 'Moth Beans',
        'type': 'Pulse / Legume',
        'season': 'Kharif (Arid/Semi-Arid)',
        'duration': '75 - 90 days',
        'description': 'Moth beans are exceptionally drought-resistant legumes grown in arid regions. They help prevent soil erosion due to their mat-like growth.',
        'tips': ['Can tolerate high temperatures and sandy soils with low fertility.', 'Avoid standing water; needs dry, sunny conditions.', 'Requires minimal nutrient inputs.']
    },
    'mungbean': {
        'name': 'Mung Bean',
        'type': 'Pulse / Legume',
        'season': 'Zaid / Kharif (Summer)',
        'duration': '60 - 75 days',
        'description': 'Mung beans are short-duration, high-protein pulses that can be grown as a catch crop between major cropping seasons.',
        'tips': ['Requires low water inputs and warm temperatures.', 'Highly responsive to phosphorous fertilization for root development.', 'Harvest when pods turn black or dark brown.']
    },
    'blackgram': {
        'name': 'Black Gram (Urad)',
        'type': 'Pulse / Legume',
        'season': 'Kharif / Zaid (Monsoon/Summer)',
        'duration': '70 - 90 days',
        'description': 'Black gram is an important pulse crop that thrives in hot and humid climates. It grows well on heavy clayey soils.',
        'tips': ['Responsive to phosphorus applications.', 'Ensure weed-free environment in initial stages.', 'Can tolerate heavier soils better than other pulses.']
    },
    'lentil': {
        'name': 'Lentil',
        'type': 'Pulse / Legume',
        'season': 'Rabi (Winter)',
        'duration': '110 - 130 days',
        'description': 'Lentils are cool-season legumes that grow well in cold climates. They are relatively drought-tolerant and prefer sandy loam soils.',
        'tips': ['Sow early in the winter season.', 'Excellent candidate for low-input agriculture.', 'Ensure effective weed management during early slow-growth phase.']
    },
    'pomegranate': {
        'name': 'Pomegranate',
        'type': 'Fruit Crop',
        'season': 'Year-round',
        'duration': '2 - 3 years to first major harvest (Perennial)',
        'description': 'Pomegranate is a hardy fruit tree adapted to dry climates. It prefers warm, dry summers and mild winters, with well-drained loamy soils.',
        'tips': ['Prune regularly to encourage sun exposure and air circulation.', 'Apply organic manure and potash during flowering and fruit set.', 'Regulate watering during fruit ripening to prevent skin splitting.']
    },
    'banana': {
        'name': 'Banana',
        'type': 'Fruit Crop',
        'season': 'Year-round (Tropical)',
        'duration': '10 - 12 months to harvest',
        'description': 'Bananas are heavy feeders requiring abundant nutrients, water, and warm tropical conditions. They thrive in deep, rich, well-drained soils.',
        'tips': ['Provide heavy applications of organic matter, nitrogen, and potassium.', 'Windbreaks are recommended as leaves tear easily in strong winds.', 'Keep the soil consistently moist but never waterlogged.']
    },
    'mango': {
        'name': 'Mango',
        'type': 'Fruit Crop',
        'season': 'Summer Harvest (Perennial tree)',
        'duration': '4 - 5 years to first harvest (Perennial)',
        'description': 'Mango is the "King of Fruits", thriving in tropical and subtropical regions. It prefers a distinct dry season to stimulate flowering.',
        'tips': ['Avoid nitrogen fertilizers during flowering to prevent vegetative flush.', 'Irrigate weekly during fruit development, then stop 2 weeks before harvest.', 'Prune dead wood and maintain center opening for light.']
    },
    'grapes': {
        'name': 'Grapes',
        'type': 'Fruit Crop (Vineyard)',
        'season': 'Spring / Summer (Perennial vine)',
        'duration': '2 - 3 years to first crop (Perennial)',
        'description': 'Grapes require a warm, dry climate during growth and ripening. They are grown on trellises and require highly specific nutrient profiles.',
        'tips': ['Pruning in winter is the most critical task for yield control.', 'Deep, infrequent watering encourages deep root systems.', 'Protect vines from fungal diseases under humid conditions.']
    },
    'watermelon': {
        'name': 'Watermelon',
        'type': 'Fruit / Gourd',
        'season': 'Zaid (Summer)',
        'duration': '80 - 100 days',
        'description': 'Watermelons are warm-season vine crops that require lots of heat, sunshine, and sandy well-drained soil to develop sweet fruits.',
        'tips': ['Provide plenty of water during vine growth, reduce watering once fruits size up.', 'Use plastic or organic mulch to warm the soil and retain moisture.', 'Bee pollination is critical; avoid spraying insecticides when flowers are open.']
    },
    'muskmelon': {
        'name': 'Muskmelon',
        'type': 'Fruit / Gourd',
        'season': 'Zaid (Summer)',
        'duration': '75 - 90 days',
        'description': 'Muskmelons are sweet summer melons. They require dry weather, warm nights, and plenty of sunlight, growing best in sandy loam soils.',
        'tips': ['Pinch off late-season blossoms so energy goes into ripening existing fruit.', 'Water deeply at the base to avoid wetting foliage.', 'Harvest when the stem "slips" or easily detaches with light pressure.']
    },
    'apple': {
        'name': 'Apple',
        'type': 'Fruit Crop',
        'season': 'Autumn Harvest (Perennial tree)',
        'duration': '3 - 5 years to fruit (Perennial)',
        'description': 'Apples are temperate fruits requiring winter chilling hours to break dormancy. They prefer well-drained loamy soils with high potassium.',
        'tips': ['Perform annual winter pruning for shape and fruit spur health.', 'Thining fruits early in summer leads to larger, high-quality apples.', 'Monitor soil pH and maintain it around 6.0-6.5.']
    },
    'orange': {
        'name': 'Orange',
        'type': 'Fruit Crop',
        'season': 'Winter / Spring Harvest (Perennial)',
        'duration': '3 - 4 years to fruit (Perennial)',
        'description': 'Oranges are evergreen citrus fruits requiring subtropical to tropical warmth. They grow best in deep, well-drained, slightly acidic soils.',
        'tips': ['Citrus trees have shallow roots; avoid deep cultivation nearby.', 'Apply nitrogen and micronutrients (Zinc, Iron) regularly.', 'Protect young trees from frost in marginal climates.']
    },
    'papaya': {
        'name': 'Papaya',
        'type': 'Fruit Crop',
        'season': 'Year-round',
        'duration': '9 - 11 months to harvest',
        'description': 'Papaya is a fast-growing, herbaceous tree-like plant. It requires warm climates, abundant sunshine, and excellent soil drainage.',
        'tips': ['Waterlogging for even 24 hours can kill papaya plants via root rot.', 'Provide regular fertilization with nitrogen and potassium.', 'Thin fruits if they are overcrowded on the trunk.']
    },
    'coconut': {
        'name': 'Coconut',
        'type': 'Fruit / Plantation',
        'season': 'Year-round (Coastal/Tropical)',
        'duration': '5 - 7 years to first fruit (Perennial)',
        'description': 'Coconuts are iconic tropical palms thriving in sandy coastal soils. They can tolerate salinity and require high humidity and regular rainfall.',
        'tips': ['Apply salt (sodium chloride) which stimulates growth in sandy soils.', 'Incorporate green manure or compost around the basin.', 'Ensure coconut palm receives direct, unobstructed sunlight.']
    },
    'cotton': {
        'name': 'Cotton',
        'type': 'Fiber Crop',
        'season': 'Kharif (Monsoon)',
        'duration': '150 - 180 days',
        'description': 'Cotton is an important industrial fiber crop. It requires a long frost-free period, moderate rainfall, and warm temperatures.',
        'tips': ['Requires deep black clayey soils (regur soil) which retain moisture.', 'Apply boron and other micronutrients for boll development.', 'Manage sucking pests and bollworms carefully during flowering.']
    },
    'jute': {
        'name': 'Jute',
        'type': 'Fiber Crop',
        'season': 'Kharif (Monsoon)',
        'duration': '120 - 150 days',
        'description': 'Jute, the "Golden Fiber", is a biodegradable plant that requires hot and humid climates and heavy rainfall. It thrives in alluvial soils.',
        'tips': ['Harvest at the flowering stage for the best quality fiber.', 'Retting (steeping in water) is crucial to extract fibers from the stem.', 'Requires clean water for high-quality retting.']
    },
    'coffee': {
        'name': 'Coffee',
        'type': 'Beverage Crop',
        'season': 'Winter Harvest (Perennial shrub)',
        'duration': '3 - 4 years to first harvest (Perennial)',
        'description': 'Coffee is grown under shade in hilly tropical regions with high rainfall. It prefers deep, acidic, organic-rich soils.',
        'tips': ['Grow under two-tier shade trees to regulate light and temperature.', 'Prune bushes regularly to maintain shape and ease harvesting.', 'Ensure soil is rich in organic humus and potassium.']
    }
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        return render_template('recommend.html')
    
    # Process POST request form inputs
    try:
        n = int(request.form.get('N'))
        p = int(request.form.get('P'))
        k = int(request.form.get('K'))
        temp = float(request.form.get('temp'))
        humidity = float(request.form.get('humidity'))
        ph = float(request.form.get('pH'))
        rainfall = float(request.form.get('rainfall'))
    except (ValueError, TypeError):
        # Return to form with error if inputs are invalid or missing
        return render_template('recommend.html', error="Please fill in all values with valid numerical entries.")

    # Create features array
    features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    
    # Predict the crop using Random Forest
    prediction_label = model.predict(features)[0]
    
    # Calculate suitability score based on classification confidence (predict_proba)
    probs = model.predict_proba(features)[0]
    classes = model.classes_
    pred_idx = list(classes).index(prediction_label)
    raw_suitability = probs[pred_idx]
    
    # Convert probability to a user-friendly percentage (e.g. 92.4%)
    # Add a slight boost based on proximity if model splits probability too much
    suitability = round(raw_suitability * 100, 1)
    
    # Ensure suitability presents correctly in the UI
    if suitability < 50.0:
        # Give a small base for the predicted class so it's not discouraging
        suitability = round(50.0 + (suitability / 2), 1)
        
    # Determine the Suitability Status
    if suitability >= 85.0:
        status = "Highly Suitable"
        status_class = "status-high"
    elif suitability >= 60.0:
        status = "Moderately Suitable"
        status_class = "status-med"
    else:
        status = "Marginally Suitable"
        status_class = "status-low"
        
    crop_info = crop_details.get(prediction_label, {
        'name': prediction_label.capitalize(),
        'type': 'Agricultural Crop',
        'season': 'Varies',
        'duration': 'N/A',
        'description': 'Details not available.',
        'tips': []
    })
    
    # Calculate diagnostics by comparing input parameters with ideal crop limits
    ideal = crop_ranges.get(prediction_label)
    diagnostics = []
    
    if ideal:
        # Nitrogen comparison
        n_min, n_max = ideal['N']
        if n < n_min:
            diagnostics.append({
                'parameter': 'Nitrogen (N)',
                'input': n,
                'ideal': f"{n_min}-{n_max}",
                'status': 'Deficient',
                'advice': 'Soil has low Nitrogen. Consider adding organic manure, compost, or a nitrogenous fertilizer like urea.',
                'icon': 'chart-bar'
            })
        elif n > n_max:
            diagnostics.append({
                'parameter': 'Nitrogen (N)',
                'input': n,
                'ideal': f"{n_min}-{n_max}",
                'status': 'Excessive',
                'advice': 'High Nitrogen levels. Avoid applying nitrogen-heavy fertilizers, as this may lead to excessive foliage but weak root systems.',
                'icon': 'chart-bar'
            })
            
        # Phosphorous comparison
        p_min, p_max = ideal['P']
        if p < p_min:
            diagnostics.append({
                'parameter': 'Phosphorus (P)',
                'input': p,
                'ideal': f"{p_min}-{p_max}",
                'status': 'Deficient',
                'advice': 'Phosphorus is deficient. Bone meal, rock phosphate, or Superphosphate are recommended to boost root growth and flowering.',
                'icon': 'chart-bar'
            })
        elif p > p_max:
            diagnostics.append({
                'parameter': 'Phosphorus (P)',
                'input': p,
                'ideal': f"{p_min}-{p_max}",
                'status': 'Excessive',
                'advice': 'High Phosphorus. Limit phosphate-heavy treatments, which can interfere with the intake of other nutrients like zinc and iron.',
                'icon': 'chart-bar'
            })
            
        # Potassium comparison
        k_min, k_max = ideal['K']
        if k < k_min:
            diagnostics.append({
                'parameter': 'Potassium (K)',
                'input': k,
                'ideal': f"{k_min}-{k_max}",
                'status': 'Deficient',
                'advice': 'Potassium is low. Apply Muriate of Potash, kelp meal, or wood ash to improve disease resistance and fruit quality.',
                'icon': 'chart-bar'
            })
        elif k > k_max:
            diagnostics.append({
                'parameter': 'Potassium (K)',
                'input': k,
                'ideal': f"{k_min}-{k_max}",
                'status': 'Excessive',
                'advice': 'High Potassium. Excess potassium can inhibit calcium and magnesium absorption in the roots.',
                'icon': 'chart-bar'
            })
            
        # pH comparison
        ph_min, ph_max = ideal['pH']
        if ph < ph_min:
            diagnostics.append({
                'parameter': 'pH Level',
                'input': ph,
                'ideal': f"{ph_min}-{ph_max}",
                'status': 'Too Acidic',
                'advice': 'Soil is too acidic. Incorporate agricultural lime (calcium carbonate) or dolomite lime to raise the pH to optimal ranges.',
                'icon': 'percentage'
            })
        elif ph > ph_max:
            diagnostics.append({
                'parameter': 'pH Level',
                'input': ph,
                'ideal': f"{ph_min}-{ph_max}",
                'status': 'Too Alkaline',
                'advice': 'Soil is too alkaline. Apply elemental sulfur, organic mulch, or peat moss to gradually lower soil pH.',
                'icon': 'percentage'
            })
            
        # Rainfall comparison
        r_min, r_max = ideal['rainfall']
        if rainfall < r_min:
            diagnostics.append({
                'parameter': 'Rainfall',
                'input': f"{rainfall} mm",
                'ideal': f"{int(r_min)}-{int(r_max)} mm",
                'status': 'Low Water',
                'advice': 'Water availability is below ideal levels. Establish supplementary drip irrigation systems to avoid crop stress.',
                'icon': 'cloud-rain'
            })
        elif rainfall > r_max:
            diagnostics.append({
                'parameter': 'Rainfall',
                'input': f"{rainfall} mm",
                'ideal': f"{int(r_min)}-{int(r_max)} mm",
                'status': 'Excess Water',
                'advice': 'High water input. Ensure adequate drainage to prevent waterlogging, soil aeration loss, and root rotting.',
                'icon': 'cloud-rain'
            })
            
        # Temperature comparison
        t_min, t_max = ideal['temp']
        if temp < t_min:
            diagnostics.append({
                'parameter': 'Temperature',
                'input': f"{temp} °C",
                'ideal': f"{int(t_min)}-{int(t_max)} °C",
                'status': 'Too Cold',
                'advice': 'Temperature is below optimal. Use mulch or row covers to protect early growth, or consider adjusting sowing time.',
                'icon': 'temperature-high'
            })
        elif temp > t_max:
            diagnostics.append({
                'parameter': 'Temperature',
                'input': f"{temp} °C",
                'ideal': f"{int(t_min)}-{int(t_max)} °C",
                'status': 'Too Hot',
                'advice': 'Temperature exceeds optimal range. Shade net usage or increased watering frequency might be necessary to mitigate heat stress.',
                'icon': 'temperature-high'
            })
            
        # Humidity comparison
        h_min, h_max = ideal['humidity']
        if humidity < h_min:
            diagnostics.append({
                'parameter': 'Humidity',
                'input': f"{humidity} %",
                'ideal': f"{int(h_min)}-{int(h_max)} %",
                'status': 'Too Dry',
                'advice': 'Air is too dry. This increases transpiration rate; monitor moisture closely.',
                'icon': 'tint'
            })
        elif humidity > h_max:
            diagnostics.append({
                'parameter': 'Humidity',
                'input': f"{humidity} %",
                'ideal': f"{int(h_min)}-{int(h_max)} %",
                'status': 'Too Humid',
                'advice': 'Excessive air humidity. Higher risk of fungal diseases. Ensure wide spacing for air circulation.',
                'icon': 'tint'
            })

    # If all parameters are optimal
    if len(diagnostics) == 0:
        diagnostics.append({
            'parameter': 'Soil & Climate Factors',
            'input': 'Optimal',
            'ideal': 'Optimal',
            'status': 'Perfect Fit',
            'advice': 'All parameters fall within the ideal growth ranges for this crop. Excellent soil health and climate setup!',
            'icon': 'check-circle'
        })
        
    input_data = {
        'N': n, 'P': p, 'K': k,
        'temp': temp, 'humidity': humidity,
        'pH': ph, 'rainfall': rainfall
    }
    
    return render_template(
        'result.html',
        crop=crop_info,
        suitability=suitability,
        status=status,
        status_class=status_class,
        diagnostics=diagnostics,
        inputs=input_data
    )

if __name__ == '__main__':
    # Start web server on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
