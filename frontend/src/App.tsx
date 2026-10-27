import { useEffect, useState } from "react";
import Card from "./components/Card";
import SectionTitle from "./components/SectionTitle";
import StatusBar from "./components/StatusBar";
import DebugStatus from "./components/DebugStatus";
import SignalChart from "./components/SignalChart";
import LiveReadoutRow from "./components/LiveReadoutRow";
import { useSensorSocket } from "./hooks/useSensorSocket";
import type { ImuData, SensorDevice } from "./types";

const MAX_POINTS = 60;
const EMPTY_IMU: ImuData = {};

function hasRealIMUData(waist: SensorDevice | null): boolean {
  return Boolean(
    waist &&
      waist.timestamp !== null &&
      waist.timestamp !== undefined &&
      waist.imu &&
      waist.imu.ax !== undefined
  );
}

export default function App() {
  const { connected, latest } = useSensorSocket();
  const [imuSeries, setImuSeries] = useState<number[]>([]);

  const waist = latest?.waist ?? null;

  useEffect(() => {
    if (!hasRealIMUData(waist) || !waist) return;

    const imu = waist.imu;
    const magnitude = Math.sqrt(
      Math.pow(Number(imu.ax ?? 0), 2) +
        Math.pow(Number(imu.ay ?? 0), 2) +
        Math.pow(Number(imu.az ?? 0), 2)
    );

    setImuSeries((prev) => {
      const next = [...prev, magnitude];
      return next.length > MAX_POINTS
        ? next.slice(next.length - MAX_POINTS)
        : next;
    });
  }, [waist]);

  const waistTimestamp = waist?.timestamp;
  const hasWaistTimestamp =
    waistTimestamp !== null && waistTimestamp !== undefined;

  const lastDevice = hasWaistTimestamp ? waist?.deviceId || "-" : "-";
  const lastTimestamp = hasWaistTimestamp
    ? String(waistTimestamp)
    : latest?.timestamp
      ? String(latest.timestamp)
      : "-";

  const waistAgeSeconds = latest?.waistImuAgeSeconds;
  const waistAge =
    waistAgeSeconds !== null && waistAgeSeconds !== undefined
      ? `${waistAgeSeconds}s ago`
      : "No waist data";

  const imuBuffer = `${latest?.imuBufferLength ?? 0}/${latest?.imuWindowSize ?? "-"}`;

  return (
    <div className="mx-auto max-w-[1320px]">
      <div className="mb-6 flex flex-col items-start gap-5 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="mb-1.5 text-[30px] tracking-[-0.02em]">
            Pre-Fall Detection
          </h1>
          <p className="text-sm text-ink-500">
            Continuous waist IMU monitoring with real-time model risk scoring.
          </p>
        </div>

        <StatusBar
          connected={connected}
          lastDevice={lastDevice}
          lastTimestamp={lastTimestamp}
        />
      </div>

      <DebugStatus waistAge={waistAge} imuBuffer={imuBuffer} />

      <Card>
        <SectionTitle title="Live Signal Overview" tag="Graph Readings" />
        <SignalChart
          title="Waist IMU Magnitude"
          data={imuSeries}
          color="#1b7bd6"
        />
      </Card>

      <LiveReadoutRow
        result={latest?.predictionResult ?? null}
        imu={waist?.imu ?? EMPTY_IMU}
      />
    </div>
  );
}
