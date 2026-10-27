from flask import Flask, abort, request, send_from_directory
from flask_socketio import SocketIO
import asyncio
import websockets
import numpy as np
import onnxruntime as ort
import json
import time
import threading
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# Flask / SocketIO
# ==========================================================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ==========================================================
# IMU FALL MODEL (Conv1D-GRU, 3 classes)
# ==========================================================
IMU_MODEL_PATH = BASE_DIR / "fall_detection.onnx"
IMU_NORM_PATH = BASE_DIR / "norm_stats.npy"

CLASS_NAMES = ["Normal", "Pre-Fall", "Fall"]
IMU_WINDOW_SIZE = 256
IMU_CHANNEL_COUNT = 6

imu_session = ort.InferenceSession(IMU_MODEL_PATH)
imu_stats = np.load(IMU_NORM_PATH, allow_pickle=True).item()
imu_mean = np.asarray(imu_stats["mean"], dtype=np.float32).reshape(-1)
imu_std = np.asarray(imu_stats["std"], dtype=np.float32).reshape(-1)

print("Loaded IMU model:", IMU_MODEL_PATH)
print("Model input shape:", imu_session.get_inputs()[0].shape)
print("Classes:", CLASS_NAMES)
print("Window size:", IMU_WINDOW_SIZE)

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
        "imu": {}
    }
}

EMIT_INTERVAL = 0.1
IMU_PREDICT_INTERVAL = 0.5


# ==========================================================
# WEBSOCKET INGEST (ESP32)
# ==========================================================
WS_INGEST_HOST = "0.0.0.0"
WS_INGEST_PORT = 8765


# ==========================================================
# REACT FRONTEND BUILD
# ==========================================================
FRONTEND_DIST = (BASE_DIR.parent / "frontend" / "dist").resolve()


# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def index():
    index_file = FRONTEND_DIST / "index.html"

    if not index_file.is_file():
        return (
            "Frontend build not found. Run 'npm install && npm run build' inside "
            "the frontend/ folder, or use the Vite dev server at "
            "http://localhost:5173",
            503,
        )

    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route("/<path:asset_path>")
def frontend_asset(asset_path):
    target = (FRONTEND_DIST / asset_path).resolve()

    if not str(target).startswith(str(FRONTEND_DIST)) or not target.is_file():
        abort(404)

    return send_from_directory(FRONTEND_DIST, asset_path)


@socketio.on("connect")
def handle_connect():
    print("SocketIO client connected:", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    print("SocketIO client disconnected:", request.sid)


# ==========================================================
# UNIT CONVERSION TO MODEL INPUT
# ==========================================================
def accel_to_model(accel_ms2):
    return float(accel_ms2)


def gyro_to_model(gyro_rads):
    return float(gyro_rads) * (180.0 / np.pi)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


# ==========================================================
# PROCESS SENSOR DATA
# ==========================================================
def process_sensor_sample(device_id, imu=None, timestamp=None):
    global imu_sample_count
    global last_waist_imu_time

    if imu is None:
        imu = {}

    # ------------------------------
    # Store latest sensor data for frontend
    # ------------------------------
    with dashboard_lock:
        if device_id not in latest_sensor_data:
            latest_sensor_data[device_id] = {
                "deviceId": device_id,
                "timestamp": None,
                "imu": {}
            }

        latest_sensor_data[device_id] = {
            "deviceId": device_id,
            "timestamp": timestamp,
            "imu": imu
        }

    # ------------------------------
    # Waist IMU CNN prediction buffer
    # ------------------------------
    if device_id == "waist_imu_01" and imu:
        try:
            last_waist_imu_time = time.time()

            ax = accel_to_model(float(imu.get("ax", 0)))
            ay = accel_to_model(float(imu.get("ay", 0)))
            az = accel_to_model(float(imu.get("az", 0)))

            gx = gyro_to_model(float(imu.get("gx", 0)))
            gy = gyro_to_model(float(imu.get("gy", 0)))
            gz = gyro_to_model(float(imu.get("gz", 0)))

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

                logits = imu_session.run(None, {"input": window.astype(np.float32)})[0][0]
                exp = np.exp(logits - np.max(logits))
                probs = exp / np.sum(exp)

                pred_index = int(np.argmax(probs))

                latest_imu_prediction = {
                    "prediction": CLASS_NAMES[pred_index],
                    "confidence": float(probs[pred_index]),
                    "normalProbability": float(probs[0]),
                    "prefallProbability": float(probs[1]),
                    "fallProbability": float(probs[2]),
                    "model": "conv1d_gru_onnx",
                    "windowSize": IMU_WINDOW_SIZE,
                    "bufferLength": buffer_length,
                    "sampleCount": current_count,
                    "status": "RUNNING"
                }

            elif window is None:
                latest_imu_prediction = {
                    "prediction": "WAITING FOR WAIST IMU DATA",
                    "confidence": None,
                    "normalProbability": None,
                    "prefallProbability": None,
                    "fallProbability": None,
                    "model": "conv1d_gru_onnx",
                    "windowSize": IMU_WINDOW_SIZE,
                    "bufferLength": buffer_length,
                    "sampleCount": current_count,
                    "status": "WAITING"
                }

        except Exception as e:
            print("IMU prediction error:", e)

            latest_imu_prediction = {
                "prediction": "IMU MODEL ERROR",
                "confidence": None,
                "normalProbability": None,
                "prefallProbability": None,
                "fallProbability": None,
                "model": "conv1d_gru_onnx",
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
# WEBSOCKET INGEST PAYLOAD
# ==========================================================
def handle_ingest_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    device_id = payload.get("deviceId") or "waist_imu_01"

    # --------------------------------------------------
    # Case 1: Batch payload
    # --------------------------------------------------
    if "samples" in payload:
        for sample in payload.get("samples", []):
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
                timestamp=sample.get("t", sample.get("timestamp"))
            )

    # --------------------------------------------------
    # Case 2: Normal single payload
    # --------------------------------------------------
    else:
        process_sensor_sample(
            device_id=device_id,
            imu=payload.get("imu", {}),
            timestamp=payload.get("timestamp")
        )


# ==========================================================
# WEBSOCKET INGEST SERVER
# ==========================================================
def ws_ingest_server():
    async def handler(connection):
        remote = getattr(connection, "remote_address", None)
        print("ESP32 WebSocket connected:", remote)

        try:
            async for message in connection:
                try:
                    handle_ingest_payload(json.loads(message))
                except Exception as e:
                    print("Error processing WebSocket message:", e)
        except Exception:
            pass
        finally:
            print("ESP32 WebSocket disconnected:", remote)

    async def serve_forever():
        async with websockets.serve(handler, WS_INGEST_HOST, WS_INGEST_PORT):
            print(f"WebSocket ingest listening on ws://{WS_INGEST_HOST}:{WS_INGEST_PORT}")
            await asyncio.Future()

    try:
        asyncio.run(serve_forever())
    except OSError as e:
        print(f"WebSocket ingest disabled, cannot bind {WS_INGEST_HOST}:{WS_INGEST_PORT}: {e}")
        print("The website still runs; ESP32 streaming is unavailable until that port is free.")


# ==========================================================
# START BACKGROUND WORKERS
# ==========================================================
threading.Thread(target=ws_ingest_server, daemon=True).start()
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