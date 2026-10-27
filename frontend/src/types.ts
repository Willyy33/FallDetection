export interface ImuData {
  ax?: number;
  ay?: number;
  az?: number;
  gx?: number;
  gy?: number;
  gz?: number;
}

export interface SensorDevice {
  deviceId: string;
  timestamp: number | string | null;
  imu: ImuData;
}

export interface PredictionResult {
  prediction: string;
  confidence: number | null;
  normalProbability: number | null;
  prefallProbability: number | null;
  fallProbability: number | null;
  model: string;
  windowSize: number;
  bufferLength?: number;
  sampleCount: number;
  status: string;
  error?: string;
}

export interface SensorUpdate {
  timestamp: number;
  waist: SensorDevice;
  predictionResult: PredictionResult | null;
  imuSampleCount: number;
  imuBufferLength: number;
  imuWindowSize: number;
  waistImuAgeSeconds: number | null;
}
