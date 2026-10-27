import Card from "./Card";
import SectionTitle from "./SectionTitle";
import type { ImuData, PredictionResult } from "../types";

const AXES: Array<{ label: string; key: keyof ImuData }> = [
  { label: "Acc X", key: "ax" },
  { label: "Acc Y", key: "ay" },
  { label: "Acc Z", key: "az" },
  { label: "Gyro X", key: "gx" },
  { label: "Gyro Y", key: "gy" },
  { label: "Gyro Z", key: "gz" },
];

const CLASSES: Array<{
  label: string;
  key: "normalProbability" | "prefallProbability" | "fallProbability";
}> = [
  { label: "Normal", key: "normalProbability" },
  { label: "Pre-Fall", key: "prefallProbability" },
  { label: "Fall", key: "fallProbability" },
];

const CELL = "rounded-panel border border-line bg-surface-soft p-4";
const CELL_LABEL = "mb-2 text-[13px] text-ink-700";
const CELL_VALUE = "text-xl font-semibold text-ink-900";

interface LiveReadoutRowProps {
  result: PredictionResult | null;
  imu: ImuData;
}

export default function LiveReadoutRow({ result, imu }: LiveReadoutRowProps) {
  const probability = result?.fallProbability;
  const hasProbability = probability !== null && probability !== undefined;

  let predictionText = "Collecting samples...";
  let predictionClass = "text-lg font-semibold text-warn";
  let fallProbabilityText = "-";

  if (hasProbability && result) {
    predictionText = result.prediction;
    fallProbabilityText = Number(probability).toFixed(4);

    if (result.prediction === "Fall") {
      predictionClass = "text-lg font-semibold text-risk";
    } else if (result.prediction === "Pre-Fall") {
      predictionClass = "text-lg font-semibold text-warn";
    } else {
      predictionClass = "text-lg font-semibold text-safe";
    }
  } else if (result?.prediction) {
    predictionText = result.prediction;
  }

  return (
    <Card>
      <SectionTitle title="Prediction & IMU Readings" tag="Live" />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-8">
        <div className={`col-span-2 ${CELL}`}>
          <h3 className={CELL_LABEL}>Waist IMU CNN</h3>
          <div className={predictionClass}>{predictionText}</div>

          <div className="mt-2 text-[13px] text-ink-500">
            Fall Probability:{" "}
            <span className="text-lg font-semibold text-ink-900">
              {fallProbabilityText}
            </span>
          </div>

          <div className="mt-3 grid grid-cols-3 gap-2 border-t border-line pt-3">
            {CLASSES.map(({ label, key }) => {
              const value = result ? result[key] : null;

              return (
                <div key={key}>
                  <div className="text-[11px] text-ink-500">{label}</div>
                  <div className="text-sm font-semibold text-ink-900">
                    {value !== null && value !== undefined
                      ? Number(value).toFixed(3)
                      : "-"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {AXES.map(({ label, key }) => (
          <div key={key} className={CELL}>
            <h3 className={CELL_LABEL}>{label}</h3>
            <div className={CELL_VALUE}>{Number(imu[key] ?? 0).toFixed(4)}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}
