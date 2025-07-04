import { useRef, useEffect } from "react";
import getColor from "@/lib/colors";

const FeatureMap = ({
  data,
  title,
  internal,
  spectrogram,
}: {
  data: number[][];
  title: string;
  internal?: boolean;
  spectrogram?: boolean;
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  if (!data || !data.length || !data[0].length) return null;

  const mapHeight = data.length;
  const mapWidth = data[0].length;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = mapWidth;
    canvas.height = mapHeight;

    const absMax = data.flat().reduce(
      (acc, val) => Math.max(acc, Math.abs(val ?? 0), 0),
      0
    );

    const imageData = ctx.createImageData(mapWidth, mapHeight);
    let idx = 0;

    for (let i = 0; i < mapHeight; i++) {
      for (let j = 0; j < mapWidth; j++) {
        const value = data[i][j];
        const normalized = absMax === 0 ? 0 : value / absMax;
        const [r, g, b] = getColor(normalized);
        imageData.data[idx++] = r;
        imageData.data[idx++] = g;
        imageData.data[idx++] = b;
        imageData.data[idx++] = 255; // Alpha
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <div className="w-full text-center">
      <canvas
        ref={canvasRef}
        className={`mx-auto block rounded border border-stone-200 ${
          internal
            ? "w-full max-w-32"
            : spectrogram
            ? "w-full object-contain"
            : "max-h-[300px] w-full max-w-[500px] object-contain"
        }`}
        style={{ imageRendering: "pixelated" }}
      />
      <p className="mt-1 text-xs text-stone-500">{title}</p>
    </div>
  );
};

export default FeatureMap;
