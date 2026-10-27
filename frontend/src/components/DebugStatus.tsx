import Card from "./Card";
import SectionTitle from "./SectionTitle";

const BOX =
  "rounded-panel border border-line bg-surface-soft p-3 text-[13px]";

interface DebugStatusProps {
  waistAge: string;
  imuBuffer: string;
}

export default function DebugStatus({ waistAge, imuBuffer }: DebugStatusProps) {
  return (
    <Card>
      <SectionTitle title="Debug Status" tag="Backend Health" />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div className={BOX}>
          <strong className="mb-1.5 block text-ink-700">Waist IMU Age</strong>
          <span>{waistAge}</span>
        </div>

        <div className={BOX}>
          <strong className="mb-1.5 block text-ink-700">IMU Buffer</strong>
          <span>{imuBuffer}</span>
        </div>
      </div>
    </Card>
  );
}
