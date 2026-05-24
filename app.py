from flask import Flask, render_template, request
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import numpy as np
import tensorflow as tf
import json
import time
import threading
from collections import deque
from pathlib import Path

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
# PRESSURE CNN MODEL - RIGHT SHOE ONLY
# ==========================================================
PRESSURE_MODEL_DIR = Path("pressure_model")

PRESSURE_MODEL_CANDIDATES = [
    PRESSURE_MODEL_DIR / "pressure_cnn_one_shoe_model_final.keras",
    PRESSURE_MODEL_DIR / "pressure_cnn_one_shoe_model.keras",
    PRESSURE_MODEL_DIR / "pressure_cnn_model_final.keras",
    PRESSURE_MODEL_DIR / "pressure_cnn_model.keras",
]

PRESSURE_MEAN_PATH = PRESSURE_MODEL_DIR / "pressure_mean.npy"
PRESSURE_STD_PATH = PRESSURE_MODEL_DIR / "pressure_std.npy"

pressure_model_path = None

for candidate in PRESSURE_MODEL_CANDIDATES:
    if candidate.exists():
        pressure_model_path = candidate
        break

if pressure_model_path is None:
    raise FileNotFoundError(
        "Pressure model not found inside /pressure_model. "
        "Expected one of: pressure_cnn_one_shoe_model_final.keras, "
        "pressure_cnn_one_shoe_model.keras, pressure_cnn_model_final.keras, "
        "pressure_cnn_model.keras"
    )

if not PRESSURE_MEAN_PATH.exists():
    raise FileNotFoundError("pressure_mean.npy not found inside /pressure_model")

if not PRESSURE_STD_PATH.exists():
    raise FileNotFoundError("pressure_std.npy not found inside /pressure_model")

pressure_model = tf.keras.models.load_model(pressure_model_path)
pressure_mean = np.load(PRESSURE_MEAN_PATH)
pressure_std = np.load(PRESSURE_STD_PATH)

pressure_input_shape = pressure_model.input_shape

if len(pressure_input_shape) != 3:
    raise ValueError(
        f"Pressure model input shape should be 3D, but got: {pressure_input_shape}"
    )

PRESSURE_WINDOW_SIZE = pressure_input_shape[1]
PRESSURE_FEATURE_COUNT = pressure_input_shape[2]

if PRESSURE_FEATURE_COUNT != 3:
    raise ValueError(
        f"Pressure model expects {PRESSURE_FEATURE_COUNT} features, "
        "but right shoe pressure input has 3 features: toe1, toe2, heel."
    )

pressure_buffer_right = deque(maxlen=PRESSURE_WINDOW_SIZE)
pressure_buffer_lock = threading.Lock()

pressure_sample_count = 0
latest_pressure_prediction = None
last_pressure_sample_count = 0
last_right_pressure_time = 0

latest_fusion_prediction = None

print("Loaded pressure model:", pressure_model_path)
print("Pressure model input shape:", pressure_model.input_shape)
print("Pressure window size:", PRESSURE_WINDOW_SIZE)
print("Pressure feature count:", PRESSURE_FEATURE_COUNT)
print("Pressure mean shape:", pressure_mean.shape)
print("Pressure std shape:", pressure_std.shape)


# ==========================================================
# DASHBOARD STATE
# ==========================================================
dashboard_lock = threading.Lock()

latest_sensor_data = {
    "waist_imu_01": {
        "deviceId": "waist_imu_01",
        "timestamp": None,
        "imu": {},
        "pressure": {},
        "topic": None
    },
    "left_shoe_01": {
        "deviceId": "left_shoe_01",
        "timestamp": None,
        "imu": {},
        "pressure": {},
        "topic": None
    },
    "right_shoe_01": {
        "deviceId": "right_shoe_01",
        "timestamp": None,
        "imu": {},
        "pressure": {},
        "topic": None
    }
}

EMIT_INTERVAL = 0.1
IMU_PREDICT_INTERVAL = 0.5
PRESSURE_PREDICT_INTERVAL = 0.5


# ==========================================================
# MQTT
# ==========================================================
MQTT_BROKER = "192.168.0.10"
MQTT_PORT = 1883

MQTT_TOPICS = [
    ("fall_detection/waist/imu", 0),
    ("fall_detection/left_shoe/data", 0),
    ("fall_detection/right_shoe/data", 0)
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

    if topic == "fall_detection/left_shoe/data":
        return "left_shoe_01"

    if topic == "fall_detection/right_shoe/data":
        return "right_shoe_01"

    return device_id or "unknown_device"


def get_pressure_value(pressure, *possible_keys):
    for key in possible_keys:
        if key in pressure:
            try:
                value = float(pressure.get(key, 0.0))
                return max(0.0, min(1.0, value))
            except Exception:
                return 0.0

    return 0.0


def extract_pressure_sample(pressure):
    toe1 = get_pressure_value(
        pressure,
        "toe1Force",
        "toe1Pressure",
        "toe_fsr402_1_pressure",
        "toe_fsr402_1"
    )

    toe2 = get_pressure_value(
        pressure,
        "toe2Force",
        "toe2Pressure",
        "toe_fsr402_2_pressure",
        "toe_fsr402_2"
    )

    heel = get_pressure_value(
        pressure,
        "heelForce",
        "heelPressure",
        "heel_rp_s40_st_pressure",
        "heel_rp_s40_st"
    )

    return [toe1, toe2, heel]


def model_scalar_output(prediction_output):
    arr = np.array(prediction_output).reshape(-1)
    return float(arr[0])


def build_fusion_prediction(imu_prediction, pressure_prediction):
    imu_prob = None
    pressure_prob = None

    if isinstance(imu_prediction, dict):
        imu_prob = imu_prediction.get("fallProbability")

    if isinstance(pressure_prediction, dict):
        pressure_prob = pressure_prediction.get("fallProbability")

    if imu_prob is None or pressure_prob is None:
        return {
            "fallProbability": None,
            "prediction": "WAITING FOR IMU + PRESSURE",
            "model": "fusion_50_50",
            "weights": {"imu": 0.5, "pressure": 0.5},
            "status": "WAITING"
        }

    fused_probability = (0.5 * float(imu_prob)) + (0.5 * float(pressure_prob))

    return {
        "fallProbability": fused_probability,
        "prediction": "PREFALL / FALL RISK" if fused_probability >= 0.5 else "NORMAL",
        "model": "fusion_50_50",
        "weights": {"imu": 0.5, "pressure": 0.5},
        "status": "RUNNING"
    }


# ==========================================================
# PROCESS SENSOR DATA
# ==========================================================
def process_sensor_sample(device_id, imu=None, pressure=None, timestamp=None, topic=None):
    global imu_sample_count
    global pressure_sample_count
    global last_right_pressure_time
    global last_waist_imu_time

    if imu is None:
        imu = {}

    if pressure is None:
        pressure = {}

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
                "pressure": {},
                "topic": None
            }

        latest_sensor_data[device_id] = {
            "deviceId": device_id,
            "timestamp": timestamp,
            "imu": imu,
            "pressure": pressure,
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

    # ------------------------------
    # Pressure CNN prediction buffer: RIGHT SHOE ONLY
    # ------------------------------
    if device_id == "right_shoe_01" and pressure:
        try:
            pressure_sample = extract_pressure_sample(pressure)

            with pressure_buffer_lock:
                pressure_buffer_right.append(pressure_sample)
                pressure_sample_count += 1
                last_right_pressure_time = time.time()

        except Exception as e:
            print("Error processing right shoe pressure sample:", e)


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
# PRESSURE PREDICTION WORKER - RIGHT SHOE ONLY
# ==========================================================
def pressure_prediction_worker():
    global latest_pressure_prediction
    global last_pressure_sample_count

    print("Pressure prediction worker started")

    while True:
        try:
            with pressure_buffer_lock:
                current_count = pressure_sample_count

                if len(pressure_buffer_right) < PRESSURE_WINDOW_SIZE:
                    window = None
                    buffer_length = len(pressure_buffer_right)
                else:
                    window = np.array(list(pressure_buffer_right), dtype=np.float32)
                    buffer_length = len(pressure_buffer_right)

            if window is not None and current_count != last_pressure_sample_count:
                last_pressure_sample_count = current_count

                expected_shape = (PRESSURE_WINDOW_SIZE, 3)

                if window.shape != expected_shape:
                    latest_pressure_prediction = {
                        "fallProbability": None,
                        "prediction": "PRESSURE WINDOW SHAPE ERROR",
                        "model": "right_shoe_pressure_cnn",
                        "windowSize": PRESSURE_WINDOW_SIZE,
                        "bufferLength": buffer_length,
                        "sampleCount": current_count,
                        "input": "right_shoe_only",
                        "status": "ERROR",
                        "error": f"Expected {expected_shape}, got {window.shape}"
                    }

                    print("Pressure window shape error:", window.shape)

                else:
                    window = (window - pressure_mean) / pressure_std

                    if window.ndim == 2:
                        window = np.expand_dims(window, axis=0)
                    elif window.ndim == 3:
                        pass
                    else:
                        raise ValueError(
                            f"Invalid pressure input ndim: {window.ndim}, shape: {window.shape}"
                        )

                    prediction_output = pressure_model.predict(window, verbose=0)
                    probability = model_scalar_output(prediction_output)

                    latest_pressure_prediction = {
                        "fallProbability": probability,
                        "prediction": "PREFALL / FALL RISK" if probability >= 0.5 else "NORMAL",
                        "model": "right_shoe_pressure_cnn",
                        "windowSize": PRESSURE_WINDOW_SIZE,
                        "bufferLength": buffer_length,
                        "sampleCount": current_count,
                        "input": "right_shoe_only",
                        "status": "RUNNING"
                    }

                    print(
                        f"Right shoe pressure prediction: {probability:.4f} | "
                        f"Samples: {current_count} | "
                        f"Buffer: {buffer_length}/{PRESSURE_WINDOW_SIZE}"
                    )

            elif window is None:
                latest_pressure_prediction = {
                    "fallProbability": None,
                    "prediction": "WAITING FOR RIGHT SHOE DATA",
                    "model": "right_shoe_pressure_cnn",
                    "windowSize": PRESSURE_WINDOW_SIZE,
                    "bufferLength": buffer_length,
                    "sampleCount": current_count,
                    "input": "right_shoe_only",
                    "status": "WAITING"
                }

        except Exception as e:
            print("Pressure prediction error:", e)

            latest_pressure_prediction = {
                "fallProbability": None,
                "prediction": "PRESSURE MODEL ERROR",
                "model": "right_shoe_pressure_cnn",
                "windowSize": PRESSURE_WINDOW_SIZE,
                "sampleCount": pressure_sample_count,
                "input": "right_shoe_only",
                "status": "ERROR",
                "error": str(e)
            }

        time.sleep(PRESSURE_PREDICT_INTERVAL)


# ==========================================================
# DASHBOARD EMIT WORKER
# ==========================================================
def dashboard_emit_worker():
    print("Dashboard emit worker started")

    while True:
        try:
            global latest_fusion_prediction

            with dashboard_lock:
                waist_data = latest_sensor_data.get("waist_imu_01", {}).copy()
                left_data = latest_sensor_data.get("left_shoe_01", {}).copy()
                right_data = latest_sensor_data.get("right_shoe_01", {}).copy()

            now = time.time()

            right_age_seconds = None
            waist_age_seconds = None

            if last_right_pressure_time:
                right_age_seconds = round(now - last_right_pressure_time, 2)

            if last_waist_imu_time:
                waist_age_seconds = round(now - last_waist_imu_time, 2)

            latest_fusion_prediction = build_fusion_prediction(
                latest_imu_prediction,
                latest_pressure_prediction
            )

            dashboard_data = {
                "timestamp": int(now * 1000),

                # Latest sensor data for frontend
                "waist": waist_data,
                "leftShoe": left_data,
                "rightShoe": right_data,

                # Predictions
                "predictionResult": latest_imu_prediction,
                "combinedPredictionResult": None,
                "pressurePredictionResult": latest_pressure_prediction,
                "fusionPredictionResult": latest_fusion_prediction,

                # IMU debug info
                "imuSampleCount": imu_sample_count,
                "imuBufferLength": len(imu_buffer),
                "imuWindowSize": IMU_WINDOW_SIZE,
                "waistImuAgeSeconds": waist_age_seconds,

                # Pressure debug info
                "pressureSampleCount": pressure_sample_count,
                "pressureBufferLength": len(pressure_buffer_right),
                "pressureWindowSize": PRESSURE_WINDOW_SIZE,
                "rightPressureAgeSeconds": right_age_seconds
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
                pressure = sample.get("pressure", None)

                if imu is None:
                    imu = {
                        "ax": sample.get("ax", 0),
                        "ay": sample.get("ay", 0),
                        "az": sample.get("az", 0),
                        "gx": sample.get("gx", 0),
                        "gy": sample.get("gy", 0),
                        "gz": sample.get("gz", 0)
                    }

                if pressure is None:
                    pressure = {
                        "toe1Force": sample.get("toe1Force", sample.get("toe1Pressure", 0)),
                        "toe2Force": sample.get("toe2Force", sample.get("toe2Pressure", 0)),
                        "heelForce": sample.get("heelForce", sample.get("heelPressure", 0))
                    }

                process_sensor_sample(
                    device_id=device_id,
                    imu=imu,
                    pressure=pressure,
                    timestamp=sample.get("t", sample.get("timestamp")),
                    topic=topic
                )

        # --------------------------------------------------
        # Case 2: Normal single payload
        # --------------------------------------------------
        else:
            imu = payload.get("imu", {})
            pressure = payload.get("pressure", {})

            process_sensor_sample(
                device_id=device_id,
                imu=imu,
                pressure=pressure,
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
threading.Thread(target=pressure_prediction_worker, daemon=True).start()
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