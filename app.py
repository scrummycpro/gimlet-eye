import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import sqlite3
from model import predict_label, train_model  # Import the model functions

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize database
def init_db():
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS labels
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       image TEXT,
                       label INTEGER)''')  # Labels are integers 1-5
    conn.commit()
    conn.close()

# Initialize the database at the start
init_db()

# Route to upload and label images (Main Page)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        label = int(request.form['label'])  # Get the label from the form
        files = request.files.getlist('files')  # Get multiple files
        
        for file in files:
            if file.filename == '':
                continue
            
            # Save each image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Save the image path and label to the database
            conn = sqlite3.connect('labels.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO labels (image, label) VALUES (?, ?)", (filepath, label))
            conn.commit()
            conn.close()

        return redirect(url_for('index'))

    return render_template('labeling.html')

# Route to classify an uploaded image
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return "No file part"
        
        if file:
            # Save the uploaded image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict the label using the model
            predicted_label = int(predict_label(filepath))

            # Map label to its description
            label_mapping = {
                1: "Sinister",
                2: "Sad/Sinister",
                3: "Sad",
                4: "Stupid",
                5: "Stupid/Sinister"
            }
            predicted_label = label_mapping.get(predicted_label, predicted_label)

            return render_template('result.html', image_url=filepath, label=predicted_label)

    return render_template('classify.html')

# Route to retrain the model
@app.route('/retrain', methods=['POST'])
def retrain():
    train_model()  # Retrain the model when the button is pressed
    return render_template('retrain_success.html')

# Explanation Page
@app.route('/explanation')
def explanation():
    return render_template('explanation.html')

if __name__ == '__main__':
    app.run(debug=True)