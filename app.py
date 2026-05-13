from flask import Flask, render_template, request
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------- Model ----------
MODEL_PATH = "prefall_model.keras"
MEAN_PATH = "mean.npy"
STD_PATH = "std.npy"

model = tf.keras.models.load_model(MODEL_PATH)
mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)

WINDOW_SIZE = 100
imu_buffer = []

# ---------- MQTT ----------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883

MQTT_TOPICS = [
    ("fall_detection/waist/imu", 0),
    ("fall_detection/left_shoe/data", 0),
    ("fall_detection/right_shoe/data", 0)
]


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("SocketIO client connected:", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    print("SocketIO client disconnected:", request.sid)

def accel_ms2_to_sisfall_raw(accel_ms2):
    # SisFall ADXL345: ±16g, 13-bit
    accel_g = accel_ms2 / 9.81
    return accel_g * 256.0


def gyro_rads_to_sisfall_raw(gyro_rads):
    # SisFall ITG3200: ±2000 deg/s, 16-bit
    gyro_deg_s = gyro_rads * (180.0 / np.pi)
    return gyro_deg_s / 0.061035

def run_prediction():
    if len(imu_buffer) < WINDOW_SIZE:
        return None

    window = np.array(imu_buffer[-WINDOW_SIZE:], dtype=np.float32)

    window = (window - mean) / std
    window = np.expand_dims(window, axis=0)

    prediction = model.predict(window, verbose=0)[0][0]

    return {
        "fallProbability": float(prediction),
        "prediction": "PREFALL / FALL RISK" if prediction >= 0.5 else "NORMAL"
    }


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(MQTT_TOPICS)
        print("Subscribed to:", MQTT_TOPICS)
    else:
        print("Failed to connect to MQTT, rc =", rc)

        


def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())

        device_id = payload.get("deviceId", "unknown_device")
        imu = payload.get("imu", {})
        pressure = payload.get("pressure", {})

        dashboard_data = {
            "topic": topic,
            "deviceId": device_id,
            "timestamp": payload.get("timestamp"),
            "imu": imu,
            "pressure": pressure,
            "predictionResult": None
        }

        # Run CNN prediction only using waist IMU
        if device_id == "waist_imu_01":
            ax = accel_ms2_to_sisfall_raw(imu.get("ax", 0))
            ay = accel_ms2_to_sisfall_raw(imu.get("ay", 0))
            az = accel_ms2_to_sisfall_raw(imu.get("az", 0))

            gx = gyro_rads_to_sisfall_raw(imu.get("gx", 0))
            gy = gyro_rads_to_sisfall_raw(imu.get("gy", 0))
            gz = gyro_rads_to_sisfall_raw(imu.get("gz", 0))

            imu_sample = [ax, ay, az, gx, gy, gz]
            imu_buffer.append(imu_sample)

            if len(imu_buffer) > WINDOW_SIZE:
                imu_buffer.pop(0)

            dashboard_data["predictionResult"] = run_prediction()

        socketio.emit("sensor_update", dashboard_data)

    except Exception as e:
        print("Error processing MQTT message:", e)


mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()


if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )