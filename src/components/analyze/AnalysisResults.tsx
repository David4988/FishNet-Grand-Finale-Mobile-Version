import { useState } from "react";
import { Check, MapPin, Calendar, Camera, Share, Download, Award, Target, Heart, Zap, Star, TrendingUp, Eye, EyeOff, Ruler } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { PredictionResult } from "@/services/tensorflow";
import { PDFGenerator } from "@/components/reports/PDFGenerator";
import { cn } from "@/lib/utils";

interface AnalysisResultsProps {
  result: PredictionResult;
  imageData: string;
  location?: { latitude: number; longitude: number };
  measuredLength?: string | number;
  onSave: () => void;
  onRetake: () => void;
  className?: string;
}

export const AnalysisResults = ({
  result,
  imageData,
  location,
  measuredLength = "1.4 m", // Default placeholder if sensor fails
  onSave,
  onRetake,
  className
}: AnalysisResultsProps) => {
  const [isSaving, setIsSaving] = useState(false);
  const [showPDFGenerator, setShowPDFGenerator] = useState(false);
  const [showMapPreview, setShowMapPreview] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onSave();
    } finally {
      setIsSaving(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-emerald-600";
    if (confidence >= 60) return "text-amber-600";
    return "text-red-500";
  };

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 80) return "bg-emerald-50 border-emerald-200";
    if (confidence >= 60) return "bg-amber-50 border-amber-200";
    return "bg-red-50 border-red-200";
  };

  const getHealthColor = (health: number) => {
    if (health >= 80) return "text-emerald-600"; // Fresh
    if (health >= 60) return "text-amber-600";   // Acceptable
    return "text-red-600";                       // Stale/Diseased
  };

  const getQualityColor = (quality: number) => {
    if (quality >= 75) return "text-emerald-600";
    if (quality >= 50) return "text-amber-600";
    return "text-red-500";
  };

  // Weighted Quality Score
  const quality = 0.6 * result.healthScore + 0.4 * result.confidence;

  const shareResults = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `FishNet Analysis: ${result.species}`,
          text: `I identified a ${result.species} (${result.estimatedWeight}kg) with FishNet AI! Quality Score: ${quality.toFixed(0)}%`,
          url: window.location.href,
        });
      } catch (error) {
        console.log('Share cancelled or failed');
      }
    } else {
      navigator.clipboard.writeText(window.location.href);
    }
  };

  // Dynamic Google Maps Embed URL based on actual GPS
  const mapUrl = location 
    ? `https://maps.google.com/maps?q=${location.latitude},${location.longitude}&t=&z=13&ie=UTF8&iwloc=&output=embed`
    : "about:blank";

  return (
    <div className={cn("space-y-6 pb-24 animate-fade-in", className)}>
      {/* 1. Fish Image Card */}
      <Card className="overflow-hidden shadow-2xl hover:shadow-glow transition-all duration-300 border-0">
        <div className="relative group">
          <img
            src={imageData}
            alt="Analyzed fish"
            className="w-full h-72 object-cover transition-transform duration-500 group-hover:scale-105"
          />
          {/* Overlay Badge */}
          <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-mono">
            AI CONFIDENCE: {result.confidence.toFixed(1)}%
          </div>
        </div>
      </Card>

      {/* 2. Species Identification Card */}
      <Card className="shadow-xl border-0 bg-gradient-card">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-3 text-xl">
            <div className="p-2 bg-gradient-primary rounded-lg shadow-sm">
              <Award className="w-5 h-5 text-white" />
            </div>
            Species Identification
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-center">
            <div className="flex items-center justify-center gap-3 mb-2">
              <Badge className="bg-gradient-primary text-white hover:opacity-90 px-3 py-1">
                <Star className="w-3 h-3 mr-1" />
                Identified Species
              </Badge>
              <Badge variant="outline" className="border-primary/30">
                Count: {result.estimatedCount}
              </Badge>
            </div>
            {/* Displays "Sardine (Mathi)" if mapped correctly in useFishNet */}
            <div className="text-3xl font-bold mt-2 text-slate-800">
              {result.species}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 3. Metrics Card (The Engine Room) */}
      <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg shadow-sm">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            AI Analysis Metrics
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          
          {/* Confidence Metric */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-blue-600" />
                <span className="font-semibold text-sm">Match Confidence</span>
              </div>
              <div className={cn("text-sm font-bold px-3 py-1 rounded-full border", getConfidenceBg(result.confidence))}>
                <span className={getConfidenceColor(result.confidence)}>
                  {result.confidence.toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="relative">
              <Progress value={result.confidence} className="h-3 bg-gray-100 rounded-full" />
              <div 
                className="absolute top-0 left-0 h-3 rounded-full bg-gradient-to-r from-blue-400 to-cyan-400 transition-all duration-1000"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>

          {/* Health/Freshness Metric */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Heart className="w-4 h-4 text-red-500" />
                <span className="font-semibold text-sm">Freshness Score</span>
              </div>
              <div className="text-sm font-bold flex gap-2 items-center">
                {/* Explicit Text Label for Clarity */}
                <span className={cn("text-xs uppercase px-2 py-0.5 rounded bg-gray-100", getHealthColor(result.healthScore))}>
                  {result.healthScore > 80 ? "Fresh" : "Stale/Risk"}
                </span>
                <span className={getHealthColor(result.healthScore)}>
                  {result.healthScore.toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="relative">
              <Progress value={result.healthScore} className="h-3 bg-gray-100 rounded-full" />
              <div 
                className="absolute top-0 left-0 h-3 rounded-full bg-gradient-to-r from-green-400 to-emerald-500 transition-all duration-1000"
                // If score is low (disease), color bar turns Red/Orange automatically via logic
                style={{ 
                    width: `${result.healthScore}%`,
                    backgroundColor: result.healthScore < 50 ? '#ef4444' : undefined 
                }}
              />
            </div>
          </div>

          {/* Weight and Quality Grid */}
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div className="bg-blue-50/50 p-4 rounded-xl border border-blue-100/50">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-blue-600" />
                <span className="text-xs font-bold text-blue-800 uppercase">Est. Weight</span>
              </div>
              <div className="text-2xl font-bold text-blue-700">
                {result.estimatedWeight.toFixed(2)} <span className="text-sm font-normal text-blue-500">kg</span>
              </div>
            </div>

            <div className="bg-emerald-50/50 p-4 rounded-xl border border-emerald-100/50">
              <div className="flex items-center gap-2 mb-2">
                <Award className="w-4 h-4 text-emerald-600" />
                <span className="text-xs font-bold text-emerald-800 uppercase">Quality</span>
              </div>
              <div className={cn("text-2xl font-bold", getQualityColor(quality))}>
                {quality.toFixed(0)}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 4. Location & Metadata Card */}
      <Card className="shadow-xl border-0 bg-white/80">
        <CardContent className="p-4">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-pink-100 rounded-lg">
                <Ruler className="w-4 h-4 text-pink-600" />
              </div>
              <div>
                <div className="font-semibold text-sm">Length</div>
                <div className="text-xs text-muted-foreground font-mono">{measuredLength}</div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Calendar className="w-4 h-4 text-purple-600" />
              </div>
              <div>
                <div className="font-semibold text-sm">Timestamp</div>
                <div className="text-xs text-muted-foreground">{new Date().toLocaleString()}</div>
              </div>
            </div>

            {location && (
              <div className="flex items-center gap-3">
                <div className="p-2 bg-teal-100 rounded-lg">
                  <MapPin className="w-4 h-4 text-teal-600" />
                </div>
                <div className="flex items-center gap-2">
                  <div>
                    <div className="font-semibold text-sm">GPS Tag</div>
                    <button
                      type="button"
                      className="text-xs text-blue-600 font-mono underline hover:text-blue-800"
                      onClick={() => setShowMapPreview(true)}
                    >
                      {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Map Modal (Cleaned up) */}
          {showMapPreview && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
              <div className="bg-white rounded-xl shadow-2xl p-4 max-w-lg w-full relative overflow-hidden">
                <button
                  className="absolute top-3 right-3 text-gray-400 hover:text-black z-10 bg-white rounded-full p-1 w-8 h-8 flex items-center justify-center shadow-md"
                  onClick={() => setShowMapPreview(false)}
                >
                  ×
                </button>
                <div className="mb-3 font-bold text-center text-lg">Catch Location</div>
                <div className="relative w-full aspect-video bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
                  <iframe 
                    className="w-full h-full absolute inset-0"
                    frameBorder="0" 
                    scrolling="no" 
                    // FIXED: Dynamic URL using actual location
                    src={mapUrl}
                    title="Catch Location"
                  ></iframe>
                </div>
                <div className="text-xs text-muted-foreground text-center mt-3 font-mono">
                  Lat: {location?.latitude} | Long: {location?.longitude}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 5. Action Buttons */}
      <div className="space-y-3">
        <div className="flex gap-3">
          <Button
            variant="outline"
            onClick={onRetake}
            className="flex-1 h-12 border-2 border-gray-200 hover:border-blue-500 hover:bg-blue-50 text-slate-700"
          >
            <Camera className="w-5 h-5 mr-2" />
            Retake
          </Button>
          
          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="flex-1 h-12 bg-gradient-primary hover:shadow-lg hover:scale-[1.02] transition-all text-white font-bold"
          >
            {isSaving ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                Saving...
              </>
            ) : (
              <>
                <Check className="w-5 h-5 mr-2" />
                Save Catch
              </>
            )}
          </Button>
        </div>

        <div className="flex gap-3">
          <Button 
            variant="secondary" 
            onClick={shareResults}
            className="flex-1 bg-blue-50 text-blue-700 hover:bg-blue-100"
          >
            <Share className="w-4 h-4 mr-2" />
            Share
          </Button>
          
          <Button 
            variant="secondary" 
            onClick={() => setShowPDFGenerator(true)}
            className="flex-1 bg-purple-50 text-purple-700 hover:bg-purple-100"
          >
            <Download className="w-4 h-4 mr-2" />
            Report
          </Button>
        </div>
      </div>

      {/* PDF Modal */}
      {showPDFGenerator && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl p-6 max-w-md w-full shadow-2xl">
            <PDFGenerator
              catch_data={{
                id: Date.now(),
                species: result.species,
                confidence: result.confidence,
                health_score: result.healthScore,
                estimated_weight: result.estimatedWeight,
                count: result.estimatedCount,
                timestamp: new Date().toISOString(),
                latitude: location?.latitude || 0,
                longitude: location?.longitude || 0,
                image_data: imageData,
                is_synced: false,
              }}
              onGenerated={() => setShowPDFGenerator(false)}
            />
            <Button 
              variant="ghost" 
              onClick={() => setShowPDFGenerator(false)}
              className="w-full mt-4 text-muted-foreground"
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};