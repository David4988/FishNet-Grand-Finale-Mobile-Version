import { useState, useRef } from "react";
import { Ruler, X, RotateCcw, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";


interface CalibrationHelperProps {
  onCalibrated: (pixelsPerCm: number, estimatedLength?: number) => void;
  onClose: () => void;
  imageData: string;
}

export const CalibrationHelper = ({ onCalibrated, onClose, imageData }: CalibrationHelperProps) => {
  const [drawing, setDrawing] = useState(false);
  const [line, setLine] = useState<{ start: { x: number; y: number } | null; end: { x: number; y: number } | null }>({ start: null, end: null });
  const [pixelsPerCm, setPixelsPerCm] = useState<number | null>(null);
  const [measuredLength, setMeasuredLength] = useState<number | null>(null);
  const [finalLine, setFinalLine] = useState<typeof line | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // Estimate pixels per cm based on image width and a typical fish size (e.g. 30cm = 80% of image width)
  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    // Assume fish is about 80% of image width and 30cm long
    const pxPerCm = (img.naturalWidth * 0.8) / 30;
    setPixelsPerCm(pxPerCm);
  };

  const reset = () => {
    setLine({ start: null, end: null });
    setFinalLine(null);
    setMeasuredLength(null);
    setDrawing(false);
  };

  // Mouse event handlers for drawing lines
  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (!drawing) {
      setLine({ start: { x, y }, end: { x, y } });
      setDrawing(true);
    } else {
      setDrawing(false);
      if (line.start && line.end && pixelsPerCm) {
        const dx = line.end.x - line.start.x;
        const dy = line.end.y - line.start.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const lengthCm = dist / pixelsPerCm;
        setMeasuredLength(lengthCm);
        setFinalLine({ ...line });
        // Do NOT call onCalibrated here, only on Continue
      }
    }
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!drawing || !line.start) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    setLine(prev => ({ ...prev, end: { x, y } }));
  };

  // Helper to render a line and label
  const renderLineWithLabel = (l: typeof line, lengthCm: number | null) => (
    l.start && l.end && pixelsPerCm ? (
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        <line
          x1={l.start.x}
          y1={l.start.y}
          x2={l.end.x}
          y2={l.end.y}
          stroke="#ef4444"
          strokeWidth={4}
          strokeDasharray="8,4"
        />
        <circle
          cx={l.start.x}
          cy={l.start.y}
          r={8}
          fill="#ef4444"
          stroke="white"
          strokeWidth={3}
        />
        <circle
          cx={l.end.x}
          cy={l.end.y}
          r={8}
          fill="#ef4444"
          stroke="white"
          strokeWidth={3}
        />
        {/* Length label */}
        <text
          x={(l.start.x + l.end.x) / 2}
          y={(l.start.y + l.end.y) / 2 - 15}
          fill="#ef4444"
          fontSize="14"
          fontWeight="bold"
          textAnchor="middle"
          className="drop-shadow-lg"
        >
          {lengthCm !== null ? lengthCm.toFixed(1) : ""} cm
        </text>
      </svg>
    ) : null
  );

  // Calculate live length while dragging
  let liveLength: number | null = null;
  if (drawing && line.start && line.end && pixelsPerCm) {
    const dx = line.end.x - line.start.x;
    const dy = line.end.y - line.start.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    liveLength = dist / pixelsPerCm;
  }

  // Continue button handler
  const handleContinue = () => {
    if (pixelsPerCm && measuredLength !== null) {
      onCalibrated(pixelsPerCm, measuredLength);
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <Card className="w-[80vw] max-h-[80vh] overflow-y-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Ruler className="w-5 h-5" />
              Size Calibration & Measurement
            </CardTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="mb-4">
            <p className="text-sm text-muted-foreground">
              Click and hold to drag a line from the fish's nose to its tail. Release to finish. The length in cm will be shown live.
            </p>
          </div>
          <div className="space-y-3">
            <div className="relative overflow-hidden rounded-lg cursor-crosshair border-2 border-primary/20 bg-black"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
            >
              <img
                ref={imageRef}
                src={imageData}
                alt="Fish for calibration"
                className="w-full h-auto max-h-96 object-contain"
                onLoad={handleImageLoad}
              />
              {/* Show persistent measured line if available */}
              {finalLine && measuredLength !== null && renderLineWithLabel(finalLine, measuredLength)}
              {/* Show live line while dragging */}
              {drawing && line.start && line.end && pixelsPerCm && renderLineWithLabel(line, liveLength)}
              {/* Instructions overlay */}
              <div className="absolute top-2 left-2 bg-black/70 text-white px-3 py-1 rounded text-sm font-medium">
                {!line.start ? "Click and hold to start measuring"
                  : drawing ? "Drag to measure fish length" : measuredLength !== null ? `Measured: ${measuredLength.toFixed(1)} cm` : "Click again to finish"}
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={reset} className="flex-1">
                <RotateCcw className="w-4 h-4 mr-2" />
                Reset
              </Button>
              <Button
                variant="default"
                className="flex-1 bg-gradient-primary text-white"
                onClick={handleContinue}
                disabled={measuredLength === null}
              >
                <Check className="w-4 h-4 mr-2" />
                Continue
              </Button>
            </div>
            <div className="bg-muted/50 p-3 rounded-lg text-xs text-muted-foreground">
              <div className="font-medium mb-1">💡 Tips for accurate measurement:</div>
              <ul className="space-y-1 list-disc list-inside">
                <li>Draw from the fish's nose to its tail</li>
                <li>Drag as straight as possible for best results</li>
                <li>Fish length is estimated based on image size</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};