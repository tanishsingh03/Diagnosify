import matplotlib
# Set the Matplotlib backend to 'Agg' to avoid Tkinter-related warnings
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, send_file
from flask_pymongo import PyMongo
import numpy as np
import pickle
import warnings
from pymongo.errors import ConnectionFailure
import tempfile
import os
import matplotlib.pyplot as plt
from fpdf import FPDF

# Suppress all warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# MongoDB Atlas Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/heart_disease_db"
mongo = PyMongo(app)

# Load the pre-trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# MongoDB Connection Test
def test_mongo_connection():
    try:
        mongo.cx.admin.command('ping')
        print("MongoDB connection successful!")
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        return False
    return True

# Check MongoDB connection when the application starts
if not test_mongo_connection():
    print("Unable to connect to MongoDB. Exiting application...")
    exit()

@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have index.html in the /templates directory

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect data from form
            input_data = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            
            # Convert to numpy array and reshape for prediction
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            # Make prediction
            prediction = model.predict(input_data_reshaped)
            prediction_result = 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have Heart Disease'

            # Store the result in MongoDB
            mongo.db.predictions.insert_one({'input_data': input_data, 'prediction': prediction_result})

            # Generate graph (you can customize this function as per your needs)
            graph_path = generate_graph(input_data)

            # Generate PDF report (you can customize this function as per your needs)
            pdf_path = generate_pdf(input_data, graph_path)

            # Return the PDF report as download
            return send_file(pdf_path, as_attachment=True, download_name='heart_disease_report.pdf')

        except Exception as e:
            return jsonify({'error': str(e)})

# Function to generate the graph
def generate_graph(input_data):
    temp_dir = tempfile.mkdtemp()

    # Create a simple bar graph for demonstration
    features = ["Age", "Sex", "Chest Pain", "Blood Pressure", "Cholesterol", "Fasting Sugar", "ECG", "Max Heart Rate", "Exercise Angina", "ST Depression", "Slope", "Vessels", "Thalassemia"]
    values = input_data

    plt.figure(figsize=(10, 6))
    plt.bar(features, values, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Heart Disease Prediction Features')
    plt.xticks(rotation=45, ha='right')

    graph_path = os.path.join(temp_dir, 'feature_graph.png')
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(graph_path)
    plt.close()

    return graph_path

# Function to generate the PDF report
def generate_pdf(input_data, graph_path):
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, 'heart_disease_report.pdf')

    pdf = FPDF()
    pdf.add_page()

    # Title with fancy design
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(0, 102, 204)  # Blue title color
    pdf.cell(200, 10, 'Heart Disease Prediction Report', ln=True, align='C')
    pdf.ln(10)

    # Introduction Section
    pdf.set_font('Arial', 'I', 12)
    pdf.set_text_color(0, 0, 0)  # Black text
    pdf.cell(200, 10, 'Introduction:', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, (
        'This report provides the results of your heart disease prediction along with detailed feature analysis. '
        'Based on your input, the prediction model has determined whether you have a risk of heart disease. '
        'Below are the details of the features provided and the model\'s prediction outcome.'
    ))
    pdf.ln(10)

    # Add input data in table format
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(90, 10, 'Feature', border=1, align='C')
    pdf.cell(90, 10, 'Value', border=1, align='C')
    pdf.ln(10)

    pdf.set_font('Arial', '', 12)
    features = ["Age", "Sex", "Chest Pain", "Blood Pressure", "Cholesterol", "Fasting Sugar", "ECG", "Max Heart Rate", "Exercise Angina", "ST Depression", "Slope", "Vessels", "Thalassemia"]
    for i, value in enumerate(input_data):
        pdf.cell(90, 10, features[i], border=1, align='C')
        pdf.cell(90, 10, f"{value}", border=1, align='C')
        pdf.ln(10)

    pdf.ln(10)

    # Add graph to the PDF
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, 'Feature Analysis Graph:', ln=True, align='L')
    pdf.ln(5)
    pdf.image(graph_path, x=10, w=180)  # Adjust size
    pdf.ln(10)

    # Prediction Outcome
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Prediction Outcome:', ln=True, align='L')
    pdf.set_font('Arial', '', 14)
    risk_percentage = calculate_risk_percentage(input_data)
    pdf.cell(200, 10, f"Risk Percentage: {risk_percentage}%", ln=True, align='C')
    pdf.ln(10)

    # Conclusion Section
    pdf.set_font('Arial', 'I', 12)
    pdf.cell(200, 10, 'Conclusion:', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, (
        'Based on the analysis, the risk of heart disease for this individual has been calculated. '
        'We recommend following up with a healthcare professional for further advice and testing.'
    ))

    pdf.output(pdf_path)

    return pdf_path

# Function to calculate the risk percentage
def calculate_risk_percentage(input_data):
    risk_percentage = model.predict_proba(np.asarray(input_data).reshape(1, -1))[0][1] * 100
    return round(risk_percentage, 2)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
