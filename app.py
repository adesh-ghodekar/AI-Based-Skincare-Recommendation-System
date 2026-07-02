from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
import pandas as pd
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from product_recommender import ProductRecommender
from skin_detection import detect_skin_type
from io import BytesIO
from database import get_db
import json
from datetime import datetime
import pytz

local_tz = pytz.timezone('Asia/Kolkata')  # change to your timezone
#timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")


app = Flask(__name__)
app.secret_key = '1234'

# Load the trained model
MODEL_PATH = "models/skin_type_model.h5"
model = load_model(MODEL_PATH)
CLASS_NAMES = ['Oily', 'Normal to Dry', 'Acne-Prone', 'Fine lines_wrinkles']

# Load product recommender
PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), 'products.csv')
recommender = ProductRecommender(PRODUCTS_PATH)

# Helper functions
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def detect_skin_type_custom(img_path, threshold=0.15):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)[0]

    oily_idx = CLASS_NAMES.index('Oily')
    dry_idx = CLASS_NAMES.index('Normal to Dry')
    is_combination = abs(predictions[oily_idx] - predictions[dry_idx]) < threshold
    primary_skin = 'Combination' if is_combination else CLASS_NAMES[np.argmax(predictions)]

    concerns = []
    for i, prob in enumerate(predictions):
        if prob > 0.3 and CLASS_NAMES[i] not in ['Oily', 'Normal to Dry']:
            concerns.append((CLASS_NAMES[i], float(prob)))

    return primary_skin, concerns, predictions

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            return "Invalid username or password"
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = get_db()
        existing_user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

        if existing_user:
            return "Username already exists. Please choose another."

        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/store_preferences', methods=['POST'])
def store_preferences():
    data = request.get_json()
    session['age'] = data.get('age')
    session['budget'] = data.get('budget')
    return '', 204


@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filepath = os.path.join('static/uploads', file.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            session['image_path'] = filepath
            return redirect(url_for('results'))
    return render_template('capture.html')

#20-4
@app.route('/analyze_skin', methods=['POST'])
def analyze_skin():
    data_url = request.form.get('image_data')
    if not data_url:
        return "No image data!", 400

    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)

    # ❌ REMOVE global timestamp
    # ✅ Generate fresh timestamp here
    timestamp = datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")

    filepath = f"static/uploads/captured_image_{timestamp}.png"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(image_data)

    session['image_path'] = filepath
    return redirect(url_for('summary'))

@app.route('/results', methods=['GET', 'POST'])
def results():
    if 'image_path' not in session:
        return redirect(url_for('capture'))

    image_path = session['image_path']
    age = session.get('age')
    budget = session.get('budget')

    skin_type, conditions, probabilities = detect_skin_type_custom(image_path)
    session['skin_type'] = skin_type
    session['conditions'] = conditions

    df = pd.read_csv(PRODUCTS_PATH)
    category_order = ['Cleanser', 'Toner', 'Serum', 'Moisturizer', 'Sunscreen', 'Face Mask']
    grouped_products = {}

    # Budget Filter Function
    def within_budget(df_):
        if budget == 'below 500':
            return df_[df_['Price'] < 500]
        elif budget == '500-1000':
            return df_[(df_['Price'] >= 500) & (df_['Price'] < 1000)]
        elif budget == '1000+':
            return df_[df_['Price'] >= 1000]
        return df_

    # Age Filterr
    age_limit = 30 if age == '30 and above' else 29 if age == '20s' else 19

    if skin_type.lower() == 'combination':
        oily = df[(df['Skin Type'].str.lower() == 'oily') & (df['Minimum Age'] <= age_limit)]
        dry = df[(df['Skin Type'].str.lower() == 'normal to dry') & (df['Minimum Age'] <= age_limit)]

        oily = within_budget(oily)
        dry = within_budget(dry)

        for cat in category_order:
            combined = pd.concat([
                oily[oily['Category'].str.lower() == cat.lower()],
                dry[dry['Category'].str.lower() == cat.lower()]
            ])
            combined = combined.drop_duplicates(subset=['Product Name'])
            sampled = combined.sample(n=min(3, len(combined))) if len(combined) >= 3 else combined
            grouped_products[cat] = sampled.to_dict(orient='records')

    else:
        filtered = df[df['Skin Type'].str.lower() == skin_type.lower()]
        filtered = filtered[(filtered['Minimum Age'] <= age_limit) | (filtered['Category'].str.lower() == 'serum')]
        filtered = within_budget(filtered)

        for cat in category_order:
            grouped_products[cat] = filtered[filtered['Category'] == cat].to_dict(orient='records')

    # Special Handling: Add wrinkle serums if "Fine lines_wrinkles" condition or Age 30+
    # After normal filtering is done

    # Special Handling: Add 1 Wrinkle serum (only if needed)
    if 'Fine lines_wrinkles' in [c[0] for c in conditions] or age == '30 and above':
        wrinkle_serums = df[
            (df['Category'].str.lower() == 'serum') & 
            (df['Skin Type'].str.lower() == 'wrinkles')
        ]

        wrinkle_serums = within_budget(wrinkle_serums)

        if not wrinkle_serums.empty:
            wrinkle_serum = wrinkle_serums.sample(n=1).iloc[0].to_dict()

            if 'Serum' not in grouped_products:
                grouped_products['Serum'] = []
            grouped_products['Serum'].append(wrinkle_serum)  # Add wrinkle serum

    return render_template('results.html', skin_type=skin_type, grouped_products=grouped_products)


import csv

# Load products from CSV once
def load_products():
    with open('products.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

all_products = load_products()

@app.route('/routine', methods=['POST'])
def routine():
    selected_ids = json.loads(request.form['selectedProducts'])

    # Build dictionary of selected products with full info
    selected_products = {}
    for category, name in selected_ids.items():
        for product in all_products:
            if product['Product Name'] == name and product['Category'] == category:
                selected_products[category] = product
                break

    # Save to session and DB
    session['selected_products'] = selected_products

    if 'username' in session:
        conn = get_db()
        current_time = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute('''
            INSERT INTO progress (username, image, skin_type, conditions, routine, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        ''', (
            session['username'],
            session.get('image_path', ''),
            session.get('skin_type', ''),
            ', '.join([c[0] for c in session.get('conditions', [])]),
            json.dumps(selected_products),
        ))
        conn.commit()

    # Render routine page
    return render_template('routine.html', selected_products=selected_products)


#20-4
@app.route('/save_selections', methods=['POST'])
def save_selections():
    selected_products = request.json.get('selected_products', {})
    session['selected_products'] = selected_products

    
    return jsonify({'success': True})



@app.route('/review', methods=['GET', 'POST'])
def review():
    if request.method == 'POST':
        review_text = request.form['review']
        with open('reviews.txt', 'a') as f:
            f.write(f"{session.get('username', 'anonymous')}: {review_text}\n")
        return redirect(url_for('index'))
    return render_template('review.html')

@app.route('/submit_review', methods=['POST'])
def submit_review():
    if 'username' not in session:
        return redirect(url_for('login'))

    review_text = request.form['review_text']
    with open('reviews.txt', 'a') as f:
        f.write(f"{session['username']}: {review_text}\n")
    return render_template('thank_you.html', review=review_text)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('welcome.html', username=session['username'])

@app.route('/summary')
def summary():
    image_path = session.get('image_path')
    if not image_path:
        return redirect(url_for('capture'))

    skin_type, conditions, probabilities = detect_skin_type_custom(image_path)
    return render_template('summary.html', 
                           skin_type=skin_type,
                           conditions=conditions,
                           probabilities=probabilities,
                           CLASS_NAMES=CLASS_NAMES)
#jhwgbdgiru34brh ogbhfvker fnenflelm\\\\\\\\\\\\\\

@app.route('/progress')
def progress():
    timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
    if 'username' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    rows = conn.execute('''
        SELECT image, skin_type, conditions, routine, timestamp
        FROM progress
        WHERE username = ?
        ORDER BY timestamp DESC
    ''', (session['username'],)).fetchall()

    progress_data = []
    unique_entries = {}

    for row in rows:
        routine = json.loads(row['routine']) if row['routine'] else {}

        # Format timestamp to readable format
        raw_ts = row['timestamp']
        try:
            naive_dt = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
            aware_dt = pytz.utc.localize(naive_dt).astimezone(local_tz)
            formatted_ts = aware_dt.strftime("%B %d, %Y")
        except:
            formatted_ts = raw_ts  # fallback if formatting fails

        if raw_ts not in unique_entries:
            unique_entries[raw_ts] = {
                'image': row['image'],
                'skin_type': row['skin_type'],
                'conditions': row['conditions'],
                'routine': routine,
                'timestamp': formatted_ts
            }

    # Convert the unique entries back to a list
    progress_data = list(unique_entries.values())

    return render_template('progress.html', progress_data=progress_data)

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M'):
    try:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').strftime(format)
    except:
        return value
    
@app.route('/skincare-tips')
def skincare_tips():
    return render_template('skincare_tips.html')



if __name__ == '__main__':
    app.run(debug=True)
