import { useState } from "react";
import { Eye, EyeOff, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { BoundingBox } from "@/types/fishnet";
import { useTranslation } from "react-i18next";

interface ExplainabilityOverlayProps {
  imageData: string;
  species: string;
  confidence: number;
  boundingBox?: BoundingBox;
  className?: string;
}

export const ExplainabilityOverlay = ({ 
  imageData, 
  species, 
  confidence, 
  boundingBox,
  className 
}: ExplainabilityOverlayProps) => {
  const { t } = useTranslation();
  const [showOverlay, setShowOverlay] = useState(true);

  // Default center box if missing
  const box = boundingBox || { yMin: 0.15, xMin: 0.15, yMax: 0.85, xMax: 0.85 };

  // � SMART POSITIONING LOGIC: If the box is too high, move the label to the bottom/right.
  const isNearTop = box.yMin < 0.15;

  const style = {
    top: `${box.yMin * 100}%`,
    left: `${box.xMin * 100}%`,
    width: `${(box.xMax - box.xMin) * 100}%`,
    height: `${(box.yMax - box.yMin) * 100}%`,
  };

  return (
    <div className={cn("relative w-full overflow-hidden bg-slate-950 shadow-2xl", className)}>
      {/* Main Image */}
      <div className="relative w-full h-full">
        <img
          src={imageData}
          alt="Fish analysis"
          className="w-full h-full object-cover" 
        />
        
        {/* Grid Texture Overlay */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay pointer-events-none"></div>
        
        {/* �️ TECH OVERLAY */}
        {showOverlay && (
          <div className="absolute inset-0">
            {/* Cinematic Vignette (Darken edges) */}
            <div className="absolute inset-0 bg-gradient-to-t from-slate-950/60 via-transparent to-slate-950/60" />
            
            {/* THE TARGET BOX */}
            <div 
              className="absolute border-2 border-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.4)] transition-all duration-700 ease-out"
              style={style}
            >
              {/* Scanline Animation */}
              <div className="absolute inset-0 overflow-hidden opacity-30">
                <div className="w-full h-full bg-gradient-to-b from-transparent via-cyan-400/20 to-transparent animate-scanline" />
              </div>

              {/* Tech Corners */}
              <div className="absolute -top-1 -left-1 w-4 h-4 border-l-2 border-t-2 border-cyan-400" />
              <div className="absolute -top-1 -right-1 w-4 h-4 border-r-2 border-t-2 border-cyan-400" />
              <div className="absolute -bottom-1 -left-1 w-4 h-4 border-l-2 border-b-2 border-cyan-400" />
              <div className="absolute -bottom-1 -right-1 w-4 h-4 border-r-2 border-b-2 border-cyan-400" />
              
              {/* �️ SMART FLOATING LABEL (Anti-Collision Logic) */}
              <div 
                className={cn(
                  "absolute flex flex-col transition-all duration-300",
                  // �️ The Fix: Avoid Top-Left (Back Button) if Near Top
                  isNearTop ? "top-2 right-2 items-end" : "-top-14 left-0 items-start" 
                )}
              >
                {/* "Target Locked" Badge */}
                <div className={cn(
                  "flex items-center gap-2 bg-slate-900/90 backdrop-blur-md border border-cyan-500/30 px-3 py-1.5 shadow-lg",
                  isNearTop ? "rounded-b-lg rounded-tl-lg" : "rounded-t-lg"
                )}>
                   <Target className="w-3 h-3 text-cyan-400 animate-pulse" />
                   <span className="text-[10px] font-mono font-bold text-cyan-100 tracking-wider uppercase">
                      {t('analyze.targetLocked')}
                   </span>
                </div>
                
                {/* Species Name Badge */}
                <div className={cn(
                  "bg-cyan-500 text-slate-950 px-3 py-1.5 text-sm font-bold shadow-lg",
                  // If near top, place this label below the badge
                  isNearTop ? "rounded-bl-lg rounded-tl-lg order-first mb-1" : "rounded-b-lg rounded-tr-lg"
                )}>
                   {species} 
                   <span className="ml-2 opacity-80 text-xs font-mono">
                     {confidence.toFixed(1)}%
                   </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Control Bar */}
      <div className="absolute bottom-4 right-4 flex gap-2 z-10">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setShowOverlay(!showOverlay)}
          className="bg-slate-900/60 backdrop-blur-md text-white hover:bg-slate-800 border border-white/10 text-xs h-8"
        >
          {showOverlay ? (
            <><EyeOff className="w-3 h-3 mr-2" /> {t('analyze.hideHud')}</>
          ) : (
            <><Eye className="w-3 h-3 mr-2" /> {t('analyze.showHud')}</>
          )}
        </Button>
      </div>
    </div>
  );
};