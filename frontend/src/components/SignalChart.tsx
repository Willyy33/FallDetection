import { useEffect, useRef } from "react";
import { drawSparkline } from "../utils/chart";

interface SignalChartProps {
  title: string;
  data: number[];
  color: string;
}

export default function SignalChart({ title, data, color }: SignalChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      drawSparkline(canvas, data, color);
    }
  }, [data, color]);

  return (
    <div className="rounded-panel border border-line bg-surface-soft p-4">
      <h3 className="mb-2.5 text-sm text-ink-700">{title}</h3>
      <canvas
        ref={canvasRef}
        width={600}
        height={160}
        className="block h-[160px] w-full"
      />
    </div>
  );
}
