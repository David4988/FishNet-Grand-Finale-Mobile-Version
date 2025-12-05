export interface BoundingBox {
  yMin: number;
  xMin: number;
  yMax: number;
  xMax: number;
}

export interface FishAnalysis {
  species: {
    name: string;
    confidence: number;
  };
  freshness: {
    score: number;
    label: "Fresh" | "Stale";
  };
  disease: {
    name: string;
    hasDisease: boolean;
    confidence: number;
  };
  boundingBox: BoundingBox;
}

export interface UIResult {
  // Core Data
  species: string;           // The internal key (e.g., "rohu")
  confidence: number;        // 0-100
  healthScore: number;       // 0-100
  disease: string;           // Internal disease name
  
  // Biometrics
  estimatedWeight: number;   // In kg
  estimatedCount: number;    // Fish count from detector
  autoLength: number;        // In cm
  boundingBox: BoundingBox;

  // Market & Env Data
  marketPrice: number;
  marketTrend: number;
  waterTemp: number;
  phLevel: number;

  // Localization (Display Strings)
  speciesLabel: string;      // "Sardine (Mathi)"
  diseaseLabel: string;      // "White Spot Risk"
  labels: Record<string, string>; // Dictionary of translated UI labels (e.g., labels.match)
}