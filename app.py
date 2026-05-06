from flask import Flask, render_template, request
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

MODEL_PATH = "prefall_model.keras"
MEAN_PATH = "mean.npy"
STD_PATH = "std.npy"

model = tf.keras.models.load_model(MODEL_PATH)
mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "fall_detection/waist_imu_01/imu"

imu_buffer = []
WINDOW_SIZE = 100


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("SocketIO client connected:", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    print("SocketIO client disconnected:", request.sid)


@app.route("/test_emit")
def test_emit():
    sample = {
        "deviceId": "test_device",
        "timestamp": "manual_test",
        "imu": {"ax": 0.1, "ay": 0.2, "az": 0.3, "gx": 0.4, "gy": 0.5, "gz": 0.6},
        "predictionResult": {"fallProbability": 0.12, "prediction": "NORMAL"}
    }
    socketio.emit("sensor_update", sample)
    return "ok"


def run_prediction():
    if len(imu_buffer) < WINDOW_SIZE:
        return None

    window = np.array(imu_buffer[-WINDOW_SIZE:])  # shape: (100, 6)

    window = (window - mean) / std
    window = np.expand_dims(window, axis=0)       # shape: (1, 100, 6)

    prediction = model.predict(window, verbose=0)[0][0]

    result = {
        "fallProbability": float(prediction),
        "prediction": "PREFALL / FALL RISK" if prediction >= 0.5 else "NORMAL"
    }

    return result


def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker")
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    try:
        # print("MQTT topic:", msg.topic)
        # print("MQTT payload:", msg.payload.decode())

        payload = json.loads(msg.payload.decode())

        ax = payload["imu"]["ax"]
        ay = payload["imu"]["ay"]
        az = payload["imu"]["az"]
        gx = payload["imu"]["gx"]
        gy = payload["imu"]["gy"]
        gz = payload["imu"]["gz"]

        imu_sample = [ax, ay, az, gx, gy, gz]
        imu_buffer.append(imu_sample)

        if len(imu_buffer) > WINDOW_SIZE:
            imu_buffer.pop(0)

        prediction_result = run_prediction()

        dashboard_data = {
            "deviceId": payload.get("deviceId", "waist_imu_01"),
            "timestamp": payload.get("timestamp"),
            "imu": payload["imu"],
            "predictionResult": prediction_result
        }

        #print("Sending to GUI:", dashboard_data)

        socketio.emit("sensor_update", dashboard_data)

    except Exception as e:
        print("Error:", e)

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