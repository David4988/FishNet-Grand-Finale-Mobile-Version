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
  measuredLength = 1.4,
  onSave,
  onRetake,
  className
}: AnalysisResultsProps) => {
  const [isSaving, setIsSaving] = useState(false);
  const [showBoundingBox, setShowBoundingBox] = useState(true);
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
    if (health >= 80) return "text-emerald-600";
    if (health >= 60) return "text-amber-600";  
    return "text-red-500";
  };

  const getQualityColor = (quality: number) => {
    if (quality >= 75) return "text-emerald-600";
    if (quality >= 50) return "text-amber-600";
    return "text-red-500";
  };

  const quality = 0.6 * result.healthScore + 0.4 * result.confidence;

  // Simulate AI bounding box coordinates (in production, this would come from the model)
  const boundingBox = {
    x: 15 + Math.random() * 20, // 15-35%
    y: 20 + Math.random() * 20, // 20-40%
    width: 50 + Math.random() * 20, // 50-70%
    height: 40 + Math.random() * 20, // 40-60%
  };

  const shareResults = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `Fish Net Analysis: ${result.species}`,
          text: `I identified a ${result.species} with ${result.confidence.toFixed(1)}% confidence using Fish Net AI!`,
          url: window.location.href,
        });
      } catch (error) {
        console.log('Share cancelled or failed');
      }
    } else {
      // Fallback to copying URL
      navigator.clipboard.writeText(window.location.href);
    }
  };

  return (
    <div className={cn("space-y-6 pb-24 animate-fade-in", className)}>
      {/* Enhanced Fish Image with prominent name overlay */}
      <Card className="overflow-hidden shadow-2xl hover:shadow-glow transition-all duration-300">
        <div className="relative group">
          <img
            src={imageData}
            alt="Analyzed fish"
            className="w-full h-72 object-cover transition-transform duration-500 group-hover:scale-105"
          />
        </div>
      </Card>

      {/* Enhanced Species Card */}
      <Card className="shadow-xl border-0 bg-gradient-card">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-3 text-xl">
            <div className="p-2 bg-gradient-primary rounded-lg">
              <Award className="w-5 h-5 text-white" />
            </div>
            Species Identification
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-center">
            <div className="flex items-center justify-center gap-3 mb-2">
              <Badge className="bg-gradient-primary text-white hover:opacity-90">
                <Star className="w-3 h-3 mr-1" />
                Identified Species
              </Badge>
              <Badge variant="outline" className="border-primary/30">
                Count: {result.estimatedCount}
              </Badge>
            </div>
            <div className="text-3xl font-bold mt-2">
              {result.species}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Analysis Metrics with Gauges */}
      <Card className="shadow-xl border-0 bg-gradient-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            AI Analysis Metrics
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Confidence with circular gauge */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-primary" />
                <span className="font-semibold">AI Confidence</span>
              </div>
              <div className={cn("text-lg font-bold px-3 py-1 rounded-full border", getConfidenceBg(result.confidence))}>
                <span className={getConfidenceColor(result.confidence)}>
                  {result.confidence.toFixed(1)}%
                </span>
              </div>
            </div>
            
            {/* Enhanced Progress Bar */}
            <div className="relative">
              <Progress 
                value={result.confidence} 
                className="h-4 bg-gray-200 rounded-full"
              />
              <div 
                className="absolute top-0 left-0 h-4 rounded-full bg-gradient-to-r from-blue-400 to-cyan-400 transition-all duration-1000"
                style={{ width: `${result.confidence}%` }}
              />
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 to-cyan-400/20 rounded-full" />
            </div>
          </div>

          {/* Health Score with enhanced visuals */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Heart className="w-4 h-4 text-red-500" />
                <span className="font-semibold">Health/Freshness</span>
              </div>
              <div className="text-lg font-bold">
                <span className={getHealthColor(result.healthScore)}>
                  {result.healthScore.toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="relative">
              <Progress 
                value={result.healthScore} 
                className="h-4 bg-gray-200 rounded-full"
              />
              <div 
                className="absolute top-0 left-0 h-4 rounded-full bg-gradient-to-r from-green-400 to-emerald-400 transition-all duration-1000"
                style={{ width: `${result.healthScore}%` }}
              />
              <div className="absolute inset-0 bg-gradient-to-r from-green-400/20 to-emerald-400/20 rounded-full" />
            </div>
          </div>

          {/* Weight and Quality in enhanced cards */}
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-xl border border-blue-100">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-800">Est. Weight</span>
              </div>
              <div className="text-2xl font-bold text-blue-600">
                {result.estimatedWeight.toFixed(1)} kg
              </div>
              <div className="text-xs text-blue-500 mt-1">
                Based on species data
              </div>
            </div>

            <div className="bg-gradient-to-br from-emerald-50 to-green-50 p-4 rounded-xl border border-emerald-100">
              <div className="flex items-center gap-2 mb-2">
                <Award className="w-4 h-4 text-emerald-600" />
                <span className="text-sm font-medium text-emerald-800">Quality Score</span>
              </div>
              <div className={cn("text-2xl font-bold", getQualityColor(quality))}>
                {quality.toFixed(0)}%
              </div>
              <div className="text-xs text-emerald-500 mt-1">
                Combined metric
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Location & Time Card */}
      <Card className="shadow-xl border-0 bg-gradient-card">
        <CardContent className="p-4">
          <div className="space-y-3">
            {/* Measured Length Display */}
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-r from-pink-500 to-yellow-500 rounded-lg">
                <Ruler className="w-4 h-4 text-white" />
              </div>
              <div>
                <div className="font-semibold">Measured Length</div>
                <div className="text-sm text-muted-foreground font-mono">
                  1.4 m
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                <Calendar className="w-4 h-4 text-white" />
              </div>
              <div>
                <div className="font-semibold">Catch Time</div>
                <div className="text-sm text-muted-foreground">
                  {new Date().toLocaleString()}
                </div>
              </div>
            </div>
            {location && (
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-green-500 to-teal-500 rounded-lg">
                  <MapPin className="w-4 h-4 text-white" />
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-semibold">GPS Location:</span>
                  <button
                    type="button"
                    className="text-sm text-muted-foreground font-mono underline hover:text-primary focus:outline-none"
                    onClick={() => setShowMapPreview(true)}
                    title="Preview location"
                  >
                    {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}
                  </button>
                </div>
              </div>
            )}
      {/* Sneak peek map modal */}
      {showMapPreview && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-white rounded-lg shadow-xl p-4 max-w-lg w-full relative">
            <button
              className="absolute top-2 right-2 text-gray-500 hover:text-black"
              onClick={() => setShowMapPreview(false)}
              aria-label="Close preview"
            >
              ×
            </button>
            <div className="mb-2 font-semibold text-center">Chennai Beach Preview</div>
            <div className="embed-map-responsive">
              <div className="embed-map-container">
                <iframe className="embed-map-frame" frameBorder="0" scrolling="no" marginHeight="0" marginWidth="0" src="https://maps.google.com/maps?width=600&height=400&hl=en&q=marina%20beach%20with%20rive&t=&z=14&ie=UTF8&iwloc=B&output=embed"></iframe>
                <a href="https://sprunkiretake.net" style={{fontSize:'2px',color:'gray',position:'absolute',bottom:0,left:0,zIndex:1,maxHeight:'1px',overflow:'hidden'}}>sprunki retake</a>
              </div>
              <style>{`.embed-map-responsive{position:relative;text-align:right;width:100%;height:0;padding-bottom:66.66666666666666%;}.embed-map-container{overflow:hidden;background:none!important;width:100%;height:100%;position:absolute;top:0;left:0;}.embed-map-frame{width:100%!important;height:100%!important;position:absolute;top:0;left:0;}`}</style>
            </div>
            <div className="text-xs text-muted-foreground text-center mt-2">Marina Beach with River</div>
          </div>
        </div>
      )}
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Action Buttons */}
      <div className="space-y-3">
        <div className="flex gap-3">
          <Button
            variant="outline"
            onClick={onRetake}
            className="flex-1 h-12 border-2 border-gray-200 hover:border-primary hover:bg-primary/5 transition-all duration-300"
          >
            <Camera className="w-5 h-5 mr-2" />
            Retake Photo
          </Button>
          
          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="flex-1 h-12 bg-gradient-primary hover:opacity-90 shadow-glow text-white font-semibold transition-all duration-300 hover:scale-105"
          >
            {isSaving ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Saving...
              </div>
            ) : (
              <>
                <Check className="w-5 h-5 mr-2" />
                Save to History
              </>
            )}
          </Button>
        </div>

        {/* Secondary Actions */}
        <div className="flex gap-3">
          <Button 
            variant="secondary" 
            onClick={shareResults}
            className="flex-1 h-11 bg-gradient-to-r from-blue-100 to-cyan-100 hover:from-blue-200 hover:to-cyan-200 text-blue-700 border-0"
          >
            <Share className="w-4 h-4 mr-2" />
            Share Catch
          </Button>
          
          <Button 
            variant="secondary" 
            onClick={() => setShowPDFGenerator(true)}
            className="flex-1 h-11 bg-gradient-to-r from-purple-100 to-pink-100 hover:from-purple-200 hover:to-pink-200 text-purple-700 border-0"
          >
            <Download className="w-4 h-4 mr-2" />
            Export PDF
          </Button>
        </div>
      </div>

      {/* PDF Generator Modal */}
      {showPDFGenerator && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-background rounded-lg p-6 max-w-md w-full">
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
              variant="outline" 
              onClick={() => setShowPDFGenerator(false)}
              className="w-full mt-4"
            >
              Close
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};