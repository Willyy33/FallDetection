from flask import Flask, render_template, request
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import numpy as np
import tensorflow as tf
import json
import time
import threading
from collections import deque

# ==========================================================
# Flask / SocketIO
# ==========================================================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ==========================================================
# IMU PREFALL MODEL
# ==========================================================
IMU_MODEL_PATH = "prefall_model.keras"
IMU_MEAN_PATH = "mean.npy"
IMU_STD_PATH = "std.npy"

imu_model = tf.keras.models.load_model(IMU_MODEL_PATH)
imu_mean = np.load(IMU_MEAN_PATH)
imu_std = np.load(IMU_STD_PATH)

IMU_WINDOW_SIZE = 100
imu_buffer = deque(maxlen=IMU_WINDOW_SIZE)
imu_buffer_lock = threading.Lock()

imu_sample_count = 0
latest_imu_prediction = None
last_imu_sample_count = 0
last_waist_imu_time = 0


# ==========================================================
# DASHBOARD STATE
# ==========================================================
dashboard_lock = threading.Lock()

latest_sensor_data = {
    "waist_imu_01": {
        "deviceId": "waist_imu_01",
        "timestamp": None,
        "imu": {},
        "topic": None
    }
}

EMIT_INTERVAL = 0.1
IMU_PREDICT_INTERVAL = 0.5


# ==========================================================
# MQTT
# ==========================================================
MQTT_BROKER = "192.168.0.10"
MQTT_PORT = 1883

MQTT_TOPICS = [
    ("fall_detection/waist/imu", 0)
]


# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("SocketIO client connected:", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    print("SocketIO client disconnected:", request.sid)


# ==========================================================
# UNIT CONVERSION FOR WAIST IMU MODEL
# ==========================================================
def accel_ms2_to_sisfall_raw(accel_ms2):
    accel_g = accel_ms2 / 9.81
    return accel_g * 256.0


def gyro_rads_to_sisfall_raw(gyro_rads):
    gyro_deg_s = gyro_rads * (180.0 / np.pi)
    return gyro_deg_s / 0.061035


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def normalize_device_id(device_id, topic):
    if topic == "fall_detection/waist/imu":
        return "waist_imu_01"

    return device_id or "unknown_device"


def model_scalar_output(prediction_output):
    arr = np.array(prediction_output).reshape(-1)
    return float(arr[0])


# ==========================================================
# PROCESS SENSOR DATA
# ==========================================================
def process_sensor_sample(device_id, imu=None, timestamp=None, topic=None):
    global imu_sample_count
    global last_waist_imu_time

    if imu is None:
        imu = {}

    device_id = normalize_device_id(device_id, topic)

    # ------------------------------
    # Store latest sensor data for frontend
    # ------------------------------
    with dashboard_lock:
        if device_id not in latest_sensor_data:
            latest_sensor_data[device_id] = {
                "deviceId": device_id,
                "timestamp": None,
                "imu": {},
                "topic": None
            }

        latest_sensor_data[device_id] = {
            "deviceId": device_id,
            "timestamp": timestamp,
            "imu": imu,
            "topic": topic
        }

    # ------------------------------
    # Waist IMU CNN prediction buffer
    # ------------------------------
    if device_id == "waist_imu_01" and imu:
        try:
            last_waist_imu_time = time.time()

            ax = accel_ms2_to_sisfall_raw(float(imu.get("ax", 0)))
            ay = accel_ms2_to_sisfall_raw(float(imu.get("ay", 0)))
            az = accel_ms2_to_sisfall_raw(float(imu.get("az", 0)))

            gx = gyro_rads_to_sisfall_raw(float(imu.get("gx", 0)))
            gy = gyro_rads_to_sisfall_raw(float(imu.get("gy", 0)))
            gz = gyro_rads_to_sisfall_raw(float(imu.get("gz", 0)))

            imu_sample = [ax, ay, az, gx, gy, gz]

            with imu_buffer_lock:
                imu_buffer.append(imu_sample)
                imu_sample_count += 1

        except Exception as e:
            print("Error processing waist IMU sample:", e)


# ==========================================================
# IMU PREDICTION WORKER
# ==========================================================
def imu_prediction_worker():
    global latest_imu_prediction
    global last_imu_sample_count

    print("IMU prediction worker started")

    while True:
        try:
            with imu_buffer_lock:
                current_count = imu_sample_count

                if len(imu_buffer) < IMU_WINDOW_SIZE:
                    window = None
                    buffer_length = len(imu_buffer)
                else:
                    window = np.array(list(imu_buffer), dtype=np.float32)
                    buffer_length = len(imu_buffer)

            if window is not None and current_count != last_imu_sample_count:
                last_imu_sample_count = current_count

                window = (window - imu_mean) / imu_std
                window = np.expand_dims(window, axis=0)

                prediction_output = imu_model.predict(window, verbose=0)
                probability = model_scalar_output(prediction_output)

                latest_imu_prediction = {
                    "fallProbability": probability,
                    "prediction": "PREFALL / FALL RISK" if probability >= 0.5 else "NORMAL",
                    "model": "waist_imu_cnn",
                    "windowSize": IMU_WINDOW_SIZE,
                    "bufferLength": buffer_length,
                    "sampleCount": current_count,
                    "status": "RUNNING"
                }

            elif window is None:
                latest_imu_prediction = {
                    "fallProbability": None,
                    "prediction": "WAITING FOR WAIST IMU DATA",
                    "model": "waist_imu_cnn",
                    "windowSize": IMU_WINDOW_SIZE,
                    "bufferLength": buffer_length,
                    "sampleCount": current_count,
                    "status": "WAITING"
                }

        except Exception as e:
            print("IMU prediction error:", e)

            latest_imu_prediction = {
                "fallProbability": None,
                "prediction": "IMU MODEL ERROR",
                "model": "waist_imu_cnn",
                "windowSize": IMU_WINDOW_SIZE,
                "sampleCount": imu_sample_count,
                "status": "ERROR",
                "error": str(e)
            }

        time.sleep(IMU_PREDICT_INTERVAL)


# ==========================================================
# DASHBOARD EMIT WORKER
# ==========================================================
def dashboard_emit_worker():
    print("Dashboard emit worker started")

    while True:
        try:
            with dashboard_lock:
                waist_data = latest_sensor_data.get("waist_imu_01", {}).copy()

            now = time.time()

            waist_age_seconds = None

            if last_waist_imu_time:
                waist_age_seconds = round(now - last_waist_imu_time, 2)

            dashboard_data = {
                "timestamp": int(now * 1000),

                # Latest sensor data for frontend
                "waist": waist_data,

                # Predictions
                "predictionResult": latest_imu_prediction,

                # IMU debug info
                "imuSampleCount": imu_sample_count,
                "imuBufferLength": len(imu_buffer),
                "imuWindowSize": IMU_WINDOW_SIZE,
                "waistImuAgeSeconds": waist_age_seconds
            }

            socketio.emit("sensor_update", dashboard_data)

        except Exception as e:
            print("SocketIO emit error:", e)

        time.sleep(EMIT_INTERVAL)


# ==========================================================
# MQTT CALLBACKS
# ==========================================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(MQTT_TOPICS)
        print("Subscribed to:", MQTT_TOPICS)
    else:
        print("Failed to connect to MQTT, rc =", rc)


def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT Broker, rc =", rc)


def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())

        if topic == "fall_detection/waist/imu":
            print("WAIST MQTT RECEIVED:", payload)

        device_id = normalize_device_id(
            payload.get("deviceId", "unknown_device"),
            topic
        )

        # --------------------------------------------------
        # Case 1: Batch payload
        # --------------------------------------------------
        if "samples" in payload:
            samples = payload.get("samples", [])

            for sample in samples:
                imu = sample.get("imu", None)

                if imu is None:
                    imu = {
                        "ax": sample.get("ax", 0),
                        "ay": sample.get("ay", 0),
                        "az": sample.get("az", 0),
                        "gx": sample.get("gx", 0),
                        "gy": sample.get("gy", 0),
                        "gz": sample.get("gz", 0)
                    }

                process_sensor_sample(
                    device_id=device_id,
                    imu=imu,
                    timestamp=sample.get("t", sample.get("timestamp")),
                    topic=topic
                )

        # --------------------------------------------------
        # Case 2: Normal single payload
        # --------------------------------------------------
        else:
            imu = payload.get("imu", {})

            process_sensor_sample(
                device_id=device_id,
                imu=imu,
                timestamp=payload.get("timestamp"),
                topic=topic
            )

    except Exception as e:
        print("Error processing MQTT message:", e)


# ==========================================================
# MQTT SETUP
# ==========================================================
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("MQTT client started")
except Exception as e:
    print("Failed to connect to MQTT broker:", e)


# ==========================================================
# START BACKGROUND WORKERS
# ==========================================================
threading.Thread(target=imu_prediction_worker, daemon=True).start()
threading.Thread(target=dashboard_emit_worker, daemon=True).start()


# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )