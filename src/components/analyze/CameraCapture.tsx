import { useState, useRef } from "react";
import { Camera, Upload, X, Check, RefreshCw, Image as ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";

interface CameraCaptureProps {
  onImageCapture: (imageData: string, file?: File) => void;
  onClose?: () => void;
  className?: string;
}

export const CameraCapture = ({ onImageCapture, onClose, className }: CameraCaptureProps) => {
  const { t } = useTranslation();
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startCamera = async () => {
    try {
      setIsLoading(true);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      fileInputRef.current?.click();
    } finally {
      setIsLoading(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (context) {
        context.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        setCapturedImage(imageData);
        
        const stream = video.srcObject as MediaStream;
        stream?.getTracks().forEach(track => track.stop());
      }
    }
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setIsLoading(true);
      const reader = new FileReader();
      reader.onload = async (e) => {
        await new Promise(resolve => setTimeout(resolve, 500));
        const imageData = e.target?.result as string;
        setCapturedImage(imageData);
        setIsLoading(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const confirmImage = () => {
    if (capturedImage) {
      onImageCapture(capturedImage);
      setCapturedImage(null);
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    startCamera();
  };

  return (
    <div className={cn("relative w-full h-full bg-black flex flex-col", className)}>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <canvas ref={canvasRef} className="hidden" />

      {!capturedImage ? (
        // --- LIVE CAMERA MODE ---
        <div className="relative flex-1 overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          
          {/* Camera UI Overlay */}
          <div className="absolute inset-0 flex flex-col justify-between p-6 z-10">
            <div className="flex justify-between items-start">
              {onClose && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClose}
                  className="bg-black/40 text-white hover:bg-black/60 rounded-full backdrop-blur-md"
                >
                  <X size={24} />
                </Button>
              )}
            </div>

            {/* Controls */}
            <div className="flex justify-around items-center pb-8">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                className="text-white/80 hover:text-white hover:bg-white/10 rounded-full w-12 h-12"
              >
                <ImageIcon size={28} />
              </Button>

              <Button
                onClick={videoRef.current ? capturePhoto : startCamera}
                disabled={isLoading}
                className="w-20 h-20 rounded-full bg-white border-4 border-white/30 shadow-[0_0_30px_rgba(255,255,255,0.3)] hover:scale-105 transition-transform flex items-center justify-center p-0"
              >
                {isLoading ? <div className="w-8 h-8 border-4 border-slate-300 border-t-transparent rounded-full animate-spin" /> : <div className="w-16 h-16 rounded-full bg-white border-2 border-slate-300" />}
              </Button>

              <div className="w-12 h-12" />
            </div>
          </div>
        </div>
      ) : (
        // --- PREVIEW MODE (Fixed) ---
        <div className="relative flex-1 bg-slate-950 flex flex-col">
          {/* Image Area - Maximized */}
          <div className="relative flex-1 overflow-hidden">
             <img
                src={capturedImage}
                alt="Preview"
                className="w-full h-full object-contain bg-black"
             />
             <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-slate-950 to-transparent pointer-events-none" />
          </div>

          {/* Control Bar */}
          <div className="bg-slate-950 p-6 pt-0 flex gap-4 items-center justify-center z-20">
            <Button
              variant="outline"
              onClick={retakePhoto}
              className="flex-1 h-14 rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10 hover:text-white backdrop-blur-md font-medium text-base"
            >
              <RefreshCw size={18} className="mr-2" />
              {t('analyze.retake')}
            </Button>
            
            <Button
              onClick={confirmImage}
              className="flex-[2] h-14 rounded-2xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-lg shadow-lg shadow-cyan-500/20"
            >
              <Check size={20} className="mr-2" />
              {t('analyze.analyzeFish')}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};