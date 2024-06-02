# interface.py
import os
import csv
import pandas as pd
import numpy as np
from math import inf
import mysql.connector
from mysql.connector import Error
from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from flask import Flask, request, jsonify
import datetime
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
import requests
from requests.exceptions import RequestException
import time
from geopy.distance import geodesic
from geopy.point import Point
import math

# Flask and related imports
flask_app = Flask(__name__)
CORS(flask_app)

connected_ip = None
app_instance = None

UPLOAD_DIRECTORY = "uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@flask_app.route('/test', methods=['GET'])
def test_endpoint():
    return 'Connection test successful'

# Endpoint to receive and save an uploaded file
@flask_app.route('/test', methods=['POST'])
def upload_file():
    global connected_ip, app_instance
    connected_ip = request.remote_addr
    
    # Ensure there's a file in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the file has a valid name
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Append a timestamp to the file name to avoid overwriting existing files
    filename, file_extension = os.path.splitext(file.filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Format: YYYYMMDDHHMMSS
    new_filename = f"{filename}_{timestamp}{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, new_filename)

    # Save the file to the upload directory
    file.save(file_path)

    # Process the file and calculate mean predictions
    try:
        csv_file_path = process_file(file_path)
        combined_df = pd.read_csv(csv_file_path)
        means = app_instance.make_predictions(combined_df)  # Use app_instance to call make_predictions
        mean_predictions_text = ", ".join([f"{mean:.2f}" for mean in means])
    except Exception as e:
        print(f"Error processing file: {e}")
        mean_predictions_text = "N/A"

    # Construct response data
    response_data = {
        "message": "File uploaded successfully",
        "file_path": file_path,
        "mean_predictions": mean_predictions_text
    }

    return jsonify(response_data), 200


@flask_app.route('/location', methods=['POST'])
def receive_location():
    try:
        data = request.form
        print(f"Received data: {data}")
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        print(f"Received location - Latitude: {latitude}, Longitude: {longitude}")

        # Call the method from outside the class
        result = app_instance.find_best_signal_within(latitude, longitude)

        if result:
            best_point, bearing, compass_direction = result
            distance = best_point['distance_m']
            signal_quality = best_point['signal_quality']
            
            print(f"Best point within .... meters:\n{best_point}")
            print(f"Direction from the given point to the best point: {bearing:.2f} degrees ({compass_direction}).")
            print(f"Signal Quality: {signal_quality}")

            app_instance.update_labels(distance, bearing, compass_direction)
            
            
            # Construct response data
            response_data = {
                "message": "Location processed successfully",
                "distance": distance,
                "direction": bearing,
                "special": compass_direction,
                "signal_quality": signal_quality
            }

            
            return jsonify(response_data), 200
        else:
            print("No suitable point found within .... meters.")
            return jsonify({"message": "No suitable point found within 30 meters."}), 404

    except Exception as e:
        print(f"Error processing location: {e}")
        return jsonify({"error": "Internal Server Error"}), 500




def map_to_weight_range(value, thresholds, weights):
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= value < thresholds[i + 1]:
            return weights[i]
    return weights[-1]

def extract_weights_mapped(X, thresholds, weights):
    mapped_weights = []
    for i in range(X.shape[1]):  # Iterate over each parameter
        parameter_values = X[:, i]
        parameter_weights = [map_to_weight_range(value, thresholds[i], weights[i]) for value in parameter_values]
        mapped_weights.append(parameter_weights)
    return np.array(mapped_weights).T

# Define thresholds and weights for each parameter
asu_thresholds = [0, 10, 20, 30, 40, 45, 50, 55, 60, 65, inf]  # Adjusted thresholds for ASU
asu_weights = [0, 0.15, 0.30, 0.39, 0.48, 0.57, 0.66, 0.75, 0.84, 0.93, 1]  # Corresponding weights for poor, fair, normal, good, and excellent

rsrp_thresholds = [-120, -115, -117, -110, -108, -105, -102, -99, -95, -90, -87, -85, -80, -75, -70, -60, inf]  # Adjusted thresholds for RSRP
rsrp_weights = [0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.67, 0.74, 0.8, 0.85, 0.9, 1]  # Corresponding weights for poor, fair, normal, good, and excellent

rsrq_thresholds = [-20, -19, -17.5, -16, -14.5, -12.5, -11, -10, -9, -8, -7, -6, -5, inf]  # Adjusted thresholds for RSRQ
rsrq_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1]  # Corresponding weights for poor, fair, normal, good, and excellent

# Combine thresholds and weights for all parameters
thresholds = [asu_thresholds, rsrp_thresholds, rsrq_thresholds]
weights = [asu_weights, rsrp_weights, rsrq_weights]

def find_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        raise FileNotFoundError(f"No files found in directory: {directory}")
    return max(files, key=os.path.getmtime)

def process_file(file_path):
    base_name = os.path.splitext(file_path)[0]
    output_csv_file = f"{base_name}.csv"

    try:
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            lines = txt_file.readlines()

        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for line in lines:
                csv_writer.writerow(line.strip().split(','))

        temp_csv_path = output_csv_file + ".tmp"
        with open(output_csv_file, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            with open(temp_csv_path, 'w', newline='', encoding='utf-8') as temp_file:
                csv_writer = csv.writer(temp_file, delimiter=',')
                for _ in range(3):
                    next(csv_reader, None)
                for row in csv_reader:
                    csv_writer.writerow(row)
        os.replace(temp_csv_path, output_csv_file)
    except PermissionError as e:
        print(f"PermissionError: {e}. Retrying in 5 seconds...")
        time.sleep(5)
        process_file(file_path)

    return output_csv_file

class FileHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            print(f"New file detected: {event.src_path}")
            self.app.root.ids.file_status_label.text = f"New file detected: {event.src_path}"
            self.app.process_latest_file(event.src_path)

class LTE_Signal(MDApp):
    def build(self):
        Window.fullscreen = True
        return Builder.load_file("main.kv")

    def on_start(self):
        global app_instance
        app_instance = self  # Store the app instance in the global variable
        flask_thread = threading.Thread(target=self.start_flask_server, daemon=True)
        flask_thread.start()
        self.start_file_observer()

    def start_flask_server(self):
        flask_app.run(host="0.0.0.0", port=5000)

    def server_running(self):
        if not hasattr(self, "dialog"):
            self.dialog = MDDialog(
                text="Server is running...",
                buttons=[
                    MDRaisedButton(text="OK", on_release=lambda x: self.dialog.dismiss())
                ],
            )
        self.dialog.open()

    def insert_into_database(self, csv_file_path):
        mydb = None
        try:
            data = pd.read_csv(csv_file_path)
            columns_to_insert = [
                "  currentTime", " longitude", " latitude", " speed", " asu", " rsrq", " rsrp"
            ]

            # Drop rows where rsrq < -200
            data = data[data[' rsrq'] >= -200]
            data = data.dropna(subset=columns_to_insert)

            # Add a new column 'signal_quality' based on the extracted weights
            columns = [' asu', ' rsrp', ' rsrq']  # Adjusted column names
            X = data[columns].values

            # Calculate mean weighted signal quality across parameters and assign to 'signal_quality' column
            data['signal_quality'] = extract_weights_mapped(X, thresholds, weights).mean(axis=1)
            mydb = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="mdrm2025#",
                database="md",
            )

            if mydb.is_connected():
                mycursor = mydb.cursor()
                insert_query = """
                INSERT INTO MyParameters (
                    currentTime, longitude, latitude, speed, asu, rsrq, rsrp, signal_quality
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """

                for _, row in data.iterrows():
                    mycursor.execute(insert_query, tuple(row[columns_to_insert + ["signal_quality"]]))

                mydb.commit()
                print("Data inserted into 'MyParameters'.")

        except Error as e:
            print(f"Error inserting data into MySQL: {e}")

        finally:
            if mydb and mydb.is_connected():
                mycursor.close()
                mydb.close()
                print("MySQL connection closed.")

    def make_predictions(self, combined_df):
        try:
            model_save_path = 'Newest_best_model.h5'
            loaded_model = keras.models.load_model(model_save_path)

            print("Model successfully loaded")

            col = [' rsrq', ' rsrp', ' asu']
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()

            selected_df = combined_df[col]
            combined_df_imputed = pd.DataFrame(imputer.fit_transform(selected_df), columns=selected_df.columns)
            combined_df_scaled = pd.DataFrame(scaler.fit_transform(combined_df_imputed), columns=selected_df.columns)

            predictions = loaded_model.predict(combined_df_scaled)

            predictions_df = pd.DataFrame(predictions, columns=['prediction'])
            combined_df_with_predictions = pd.concat([combined_df, predictions_df], axis=1)

            combined_df_with_predictions.to_csv('combined_predictions.csv', index=False)
            print("Predictions saved to 'combined_predictions.csv'.")

            # Display that predictions are saved
            self.root.ids.prediction_status_label.text = "Predictions saved to 'combined_predictions.csv'."

            # Calculate and display the means
            num_rows = combined_df_with_predictions.shape[0]
            means = combined_df_with_predictions['prediction'].groupby(
                combined_df_with_predictions.index // (num_rows // 1)).mean()
            means_text = ", ".join([f"{mean:.2f}" for mean in means])
            self.root.ids.prediction_means_label.text = f"Mean Predictions: {means_text}"
            return means

        except Exception as e:
            print(f"Error making predictions: {e}")

    def process_latest_file(self, file_path):
        try:
            csv_file_path = process_file(file_path)
            self.insert_into_database(csv_file_path)
            combined_df = pd.read_csv(csv_file_path)
            self.make_predictions(combined_df)
        except Exception as e:
            print(f"Error processing latest file: {e}")

    
    def calculate_initial_compass_bearing(self,pointA, pointB):
        """
        Calculates the initial compass bearing between two points.
        """
        lat1 = math.radians(pointA.latitude)
        lat2 = math.radians(pointB.latitude)
        diff_long = math.radians(pointB.longitude - pointA.longitude)

        x = math.sin(diff_long) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_long)

        initial_bearing = math.atan2(x, y)

        # Convert radians to degrees and normalize to 0-360 degrees
        compass_bearing = (math.degrees(initial_bearing) + 360) % 360

        return compass_bearing
    
    def connect_to_db(self):
        # Implement your database connection here
        try:
            mydb = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="mdrm2025#",
                database="md",
            )
            return mydb
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None
    
    
    def find_best_signal_within(self, lat, lon):
        mydb = self.connect_to_db()
        if not mydb:
            return None

        try:
            mycursor = mydb.cursor()

            # Fetch all records from the Parameters table
            query = "SELECT currentTime, longitude, latitude, signal_quality FROM MyParameters"
            mycursor.execute(query)
            results = mycursor.fetchall()

            # Convert to a DataFrame
            columns = ['currentTime', 'longitude', 'latitude', 'signal_quality']
            data = pd.DataFrame(results, columns=columns)

            # Calculate the distance from the given point for each row
            given_point = Point(lat, lon)
            data['distance_m'] = data.apply(lambda row: geodesic(given_point, Point(row['latitude'], row['longitude'])).meters, axis=1)

            # Filter for points within 30 meters
            nearby_points = data[data['distance_m'] <= 30]

            if nearby_points.empty:
                print("No points found within close meters.")
                return None

            # Find the point with the best signal quality (based on `signal_quality`)
            best_point = nearby_points.loc[nearby_points['signal_quality'].idxmax()]

            # Calculate the compass bearing from the given point to the best point
            best_point_position = Point(best_point['latitude'], best_point['longitude'])
            compass_bearing = self.calculate_initial_compass_bearing(given_point, best_point_position)

            # Direction description for compass bearing
            def get_compass_direction(bearing):
                if 0 <= bearing < 45 or 315 <= bearing <= 360:
                    return "North"
                elif 45 <= bearing < 135:
                    return "East"
                elif 135 <= bearing < 225:
                    return "South"
                elif 225 <= bearing < 315:
                    return "West"
                else:
                    return f"{bearing:.2f} degrees"

            compass_direction = get_compass_direction(compass_bearing)

            return best_point, compass_bearing, compass_direction

        except Exception as e:
            print(f"Error processing signal data: {e}")
            return None
        finally:
            mycursor.close()
            mydb.close()
    
    def update_labels(self, distance, bearing, compass_direction):
        self.root.ids.distance_label.text = f"Distance: {distance:.2f} meters"
        self.root.ids.direction_label.text = f"Direction: {bearing:.2f} degrees ({compass_direction})"
    
    def start_file_observer(self):
        self.observer = Observer()
        self.event_handler = FileHandler(self)
        self.observer.schedule(self.event_handler, UPLOAD_DIRECTORY, recursive=False)
        self.observer.start()

if __name__ == '__main__':
    LTE_Signal().run()
