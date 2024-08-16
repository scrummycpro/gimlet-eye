import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import sqlite3
from model import predict_label  # Import the prediction function from model.py

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

# Route to upload and label images
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
            predicted_label = predict_label(filepath)

            # Map label '1' to 'Kali'
            if predicted_label == 1:
                predicted_label = "Kali"
            else:
                predicted_label = int(predicted_label)  # Convert to integer for other labels

            return render_template('result.html', image_url=filepath, label=predicted_label)

    return render_template('classify.html')

if __name__ == '__main__':
    app.run(debug=True)