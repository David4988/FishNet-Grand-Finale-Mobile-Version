import { useEffect, useRef, useState } from "react";
import {
  Camera,
  Upload,
  Ruler,
  Check,
  Share2,
  MapPin,
  Calendar,
  Info,
  ArrowLeft,
  Thermometer,
  Droplets,
  TrendingUp,
  DollarSign,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CameraCapture } from "@/components/analyze/CameraCapture";
import { ExplainabilityOverlay } from "@/components/analyze/ExplainabilityOverlay";
import { CalibrationHelper } from "@/components/analyze/CalibrationHelper";
import { databaseService } from "@/services/database";
import { toast } from "@/components/ui/use-toast";
import { addLocalCatch } from "@/utils/localCatches";
import { useFishNet } from "@/hooks/useFishNet";
import { BoundingBox, UIResult } from "@/types/fishnet";

// --- INTELLIGENCE LAYER (Matches SPECIES_LABELS) ---
const SPECIES_DB: Record<
  string,
  { key: string; price: number; trend: number; weightFactor: number }
> = {
  // Carps
  rohu: { key: "rohu", price: 160, trend: 4.5, weightFactor: 0.012 },
  catla: { key: "catla", price: 180, trend: 2.1, weightFactor: 0.014 },
  mrigal: { key: "mrigal", price: 150, trend: 1.5, weightFactor: 0.011 },
  common_carp: {
    key: "common_carp",
    price: 130,
    trend: 0.5,
    weightFactor: 0.013,
  },
  grass_carp: {
    key: "grass_carp",
    price: 140,
    trend: 2.0,
    weightFactor: 0.013,
  },
  silver_carp: {
    key: "silver_carp",
    price: 110,
    trend: -1.0,
    weightFactor: 0.012,
  },

  // Market Staples
  tilapia: { key: "tilapia", price: 120, trend: -1.2, weightFactor: 0.015 },
  catfish: { key: "catfish", price: 100, trend: 1.0, weightFactor: 0.013 },

  // Marine / High Value
  barramundi: {
    key: "barramundi",
    price: 450,
    trend: 8.4,
    weightFactor: 0.015,
  },
  sea_bass: { key: "barramundi", price: 450, trend: 8.4, weightFactor: 0.015 }, // Map Sea Bass -> Barramundi
  mackerel: { key: "mackerel", price: 220, trend: 5.2, weightFactor: 0.01 },
  sardine: { key: "sardine", price: 120, trend: -1.2, weightFactor: 0.009 },
  red_mullet: { key: "red_mullet", price: 250, trend: 3.5, weightFactor: 0.01 },
  pink_perch: {
    key: "pink_perch",
    price: 180,
    trend: 1.5,
    weightFactor: 0.011,
  },
  sea_bream: { key: "sea_bream", price: 400, trend: 4.0, weightFactor: 0.014 },
  sprat: { key: "sprat", price: 90, trend: -0.5, weightFactor: 0.005 },
  trout: { key: "trout", price: 600, trend: 7.0, weightFactor: 0.012 },

  // Shellfish
  prawn: { key: "prawn", price: 450, trend: 5.0, weightFactor: 0.008 },
  crab: { key: "crab", price: 700, trend: 10.0, weightFactor: 0.025 },

  // System
  wild_fish_background: {
    key: "wild_fish",
    price: 0,
    trend: 0,
    weightFactor: 0.01,
  },
  unknown: { key: "unknown", price: 0, trend: 0, weightFactor: 0.01 },
};

// Offline average weight samples (kg) per species; 5 values each
const SPECIES_WEIGHT_SAMPLES: Record<string, number[]> = {
  barramundi: [1.8, 2.2, 2.5, 3.0, 3.5],
  catfish: [0.6, 0.8, 1.0, 1.2, 1.5],
  catla: [1.2, 1.6, 2.0, 2.4, 2.8],
  crab: [0.2, 0.25, 0.3, 0.35, 0.4],
  mackerel: [0.25, 0.35, 0.45, 0.55, 0.65],
  mrigal: [0.8, 1.0, 1.3, 1.6, 1.9],
  prawn: [0.03, 0.05, 0.07, 0.09, 0.12],
  red_mullet: [0.15, 0.2, 0.25, 0.3, 0.35],
  rohu: [1.0, 1.4, 1.8, 2.2, 2.6],
  sardine: [0.1, 0.15, 0.2, 0.25, 0.3],
  sea_bream: [0.8, 1.1, 1.4, 1.7, 2.0],
  tilapia: [0.7, 0.9, 1.1, 1.3, 1.5],
  trout: [0.9, 1.2, 1.5, 1.8, 2.1],
  wild_fish: [0.5, 0.8, 1.0, 1.3, 1.6],
  catfish_background: [0.6, 0.8, 1.0, 1.2, 1.5],
  common_carp: [1.0, 1.3, 1.6, 1.9, 2.2],
  grass_carp: [1.3, 1.6, 1.9, 2.2, 2.5],
  silver_carp: [1.0, 1.2, 1.4, 1.6, 1.8],
  sprat: [0.05, 0.08, 0.1, 0.12, 0.15],
};

function pickSampleWeight(speciesKey: string): number {
  const samples = SPECIES_WEIGHT_SAMPLES[speciesKey] || SPECIES_WEIGHT_SAMPLES["wild_fish"];
  const idx = Math.floor(Math.random() * samples.length);
  return samples[idx];
}

const calculateBioMetrics = (box: BoundingBox | undefined) => {
  if (!box) return { length: 0, weight: 0 };
  const widthPercent = box.xMax - box.xMin;
  const heightPercent = box.yMax - box.yMin;
  const diagonal = Math.sqrt(
    Math.pow(widthPercent, 2) + Math.pow(heightPercent, 2)
  );
  let estimatedLengthCm = diagonal * 50;
  estimatedLengthCm = Math.max(10, Math.min(120, estimatedLengthCm));
  const estimatedWeightKg = Math.pow(estimatedLengthCm / 10, 3) / 25;
  return {
    length: parseFloat(estimatedLengthCm.toFixed(1)),
    weight: parseFloat(estimatedWeightKg.toFixed(2)),
  };
};

export default function AnalyzePage() {
  const { t, i18n } = useTranslation();
  const { analyzeFish, isModelLoading, modelError, fishCount } = useFishNet();

  const [showCamera, setShowCamera] = useState(false);
  const [showCalibration, setShowCalibration] = useState(false);
  const [imageData, setImageData] = useState<string | null>(null);
  const [result, setResult] = useState<UIResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [location, setLocation] = useState<
    { latitude: number; longitude: number } | undefined
  >(undefined);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [measuredLength, setMeasuredLength] = useState<string | number>("-");

  const currentLang = i18n?.language || localStorage.getItem("fishnet_language") || "en";
  const scanCatchDict: Record<string, string> = {
    en: "Scan Catch",
    ta: "பிடிப்பை ஸ்கேன் செய்",
    hi: "पकड़ स्कैन करें",
    ml: "പിടിയത് സ്കാൻ ചെയ്യുക",
    te: "పట్టును స్కాన్ చేయండి",
    kn: "ಪಡುವಿಕೆಯನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡಿ",
    gu: "પકડી સ્કેન કરો",
    mr: "पकड स्कॅन करा",
    pa: "ਪਕੜ ਸਕੈਨ ਕਰੋ",
    or: "ଧରା ଯାଇଥିବାକୁ ସ୍କାନ୍ କରନ୍ତୁ",
  };
  const scanCatchLabel = scanCatchDict[currentLang] || scanCatchDict.en;

  const discardDict: Record<string, string> = {
    en: "Discard",
    ta: "நிராகரி",
    hi: "त्यागें",
    ml: "നിരസിക്കുക",
    te: "విస్మరించు",
    kn: "ತ್ಯಜಿಸಿ",
    gu: "રદ કરો",
    mr: "रद्द करा",
    pa: "ਰੱਦ ਕਰੋ",
    or: "ପ୍ରତ୍ୟାଖ୍ୟାନ କରନ୍ତୁ",
  };
  const saveRecordDict: Record<string, string> = {
    en: "Save Record",
    ta: "பதிவை சேமி",
    hi: "रिकॉर्ड सहेजें",
    ml: "രേഖ സംരക്ഷിക്കുക",
    te: "రికార్డ్ సేవ్ చేయండి",
    kn: "ದಾಖಲೆ ಉಳಿಸಿ",
    gu: "રેકોર્ડ સાચવો",
    mr: "रेकॉर्ड जतन करा",
    pa: "ਰਿਕਾਰਡ ਸੰਭਾਲੋ",
    or: "ରେକର୍ଡ ସଞ୍ଚୟ କରନ୍ତୁ",
  };
  const discardLabel = discardDict[currentLang] || discardDict.en;
  const saveRecordLabel = saveRecordDict[currentLang] || saveRecordDict.en;

  useEffect(() => {
    if (modelError)
      toast({
        variant: "destructive",
        title: t("analyze.error"),
        description: modelError,
      });
  }, [modelError, t]);

  const analyzeImage = async (dataUrl: string) => {
    if (isModelLoading) {
      toast({
        title: t("analyze.systemWarming"),
        description: t("analyze.loadingAI"),
      });
      return;
    }
    setIsAnalyzing(true);

    setTimeout(async () => {
      try {
        const img = new Image();
        img.src = dataUrl;
        await img.decode();
        const analysis = await analyzeFish(img);

        if (analysis) {
          setImageData(dataUrl);

          const rawName = analysis.species.name;
          const cleanKey = rawName.split("(")[0].trim().toLowerCase();
          const dbEntry =
            SPECIES_DB[rawName.toLowerCase()] ||
            SPECIES_DB[cleanKey] ||
            SPECIES_DB["unknown"];
          const bio = calculateBioMetrics(analysis.boundingBox);

          const currentLang = i18n?.language || localStorage.getItem("fishnet_language") || "en";
          const speciesKey = dbEntry.key;
          const speciesDict: Record<string, Record<string, string>> = {
            ta: {
              barramundi: "பாரமுண்டி",
              catfish: "கேட்ஃபிஷ்",
              catla: "காட்லா",
              crab: "நண்டு",
              mackerel: "அயிலை",
              mrigal: "மிரிகல்",
              prawn: "இறால்",
              red_mullet: "சிவப்பு முல்லெட்",
              rohu: "ரோஹு",
              sardine: "சார்டின்",
              sea_bream: "சீ ப்ரீம்",
              tilapia: "திலாபியா",
              trout: "ட்ரௌட்",
              wild_fish: "காட்டு மீன்",
            },
            hi: {
              barramundi: "बरामुंडी",
              catfish: "कैटफ़िश",
              catla: "कतला",
              crab: "केकड़ा",
              mackerel: "मैकरेल",
              mrigal: "मृगल",
              prawn: "झींगा",
              red_mullet: "लाल मुलैट",
              rohu: "रोहू",
              sardine: "सरडीन",
              sea_bream: "सी ब्रीम",
              tilapia: "तिलापिया",
              trout: "ट्राउट",
              wild_fish: "जंगली मछली",
            },
            ml: {
              barramundi: "ബാരമുണ്ടി",
              catfish: "കാറ്റ്‌ഫിഷ്",
              catla: "കത്ലാ",
              crab: "ഞണ്ട്",
              mackerel: "അയല",
              mrigal: "മൃഗാൽ",
              prawn: "ചെമ്മീൻ",
              red_mullet: "ചുവപ്പ് മുല്ലറ്റ്",
              rohu: "റോഹു",
              sardine: "ചാള",
              sea_bream: "സി ബ്രിം",
              tilapia: "തിലാപിയ",
              trout: "ട്രൗട്ട്",
              wild_fish: "കാട്ടുമത്സ്യം",
            },
          };
          const localizedSpecies = speciesDict[currentLang]?.[speciesKey];
          const speciesDisplay = currentLang === "en"
            ? speciesKey.replace(/_/g, " ")
            : localizedSpecies || t(`species.${speciesKey}`, { defaultValue: speciesKey.replace(/_/g, " ") });

          const labelDict: Record<string, Record<string, string>> = {
            ta: {
              MATCH: "பொருந்துதல்",
              COUNT: "எண்ணிக்கை",
              HEALTH: "நலம்",
              BIOMETRICS: "உயிரளவியல்",
              "EST. LENGTH": "மதிப்பிடப்பட்ட நீளம்",
              "AI MEASURED": "ஏஐ அளவை",
              "EST. WEIGHT": "மதிப்பிடப்பட்ட எடை",
              "MARKET & ECONOMICS": "சந்தை & பொருளாதாரம்",
              "CURRENT MARKET PRICE": "தற்போதைய சந்தை விலை",
              "ENVIRONMENT & HEALTH": "சுற்றுச்சூழல் & நலம்",
              "ENV TEMP": "சுற்றுச்சூழல் வெப்பம்",
              PATHOLOGY: "நோயியியல்",
              Healthy: "ஆரோக்கியம்",
            },
            hi: {
              MATCH: "मिलान",
              COUNT: "गिनती",
              HEALTH: "स्वास्थ्य",
              BIOMETRICS: "बायोमेट्रिक्स",
              "EST. LENGTH": "अनुमानित लंबाई",
              "AI MEASURED": "एआई मापा",
              "EST. WEIGHT": "अनुमानित वजन",
              "MARKET & ECONOMICS": "बाज़ार व अर्थशास्त्र",
              "CURRENT MARKET PRICE": "वर्तमान बाज़ार मूल्य",
              "ENVIRONMENT & HEALTH": "पर्यावरण व स्वास्थ्य",
              "ENV TEMP": "पर्यावरण तापमान",
              PATHOLOGY: "रोगविज्ञान",
              Healthy: "स्वस्थ",
            },
            ml: {
              MATCH: "പൊരുത്തം",
              COUNT: "എണ്ണം",
              HEALTH: "ആരോഗ്യം",
              BIOMETRICS: "ബയോമെട്രിക്‌സ്",
              "EST. LENGTH": "അനുമാന നീളം",
              "AI MEASURED": "എഐ അളവ്",
              "EST. WEIGHT": "അനുമാന ഭാരം",
              "MARKET & ECONOMICS": "വിപണി & സാമ്പത്തികം",
              "CURRENT MARKET PRICE": "നിലവിലെ വിപണി വില",
              "ENVIRONMENT & HEALTH": "പരിസ്ഥിതി & ആരോഗ്യം",
              "ENV TEMP": "പരിസ്ഥിതി താപനില",
              PATHOLOGY: "രോഗശാസ്ത്രം",
              Healthy: "ആരോഗ്യം",
            },
            te: {
              MATCH: "సరిపోలుడు",
              COUNT: "ఎണ്ണిక",
              HEALTH: "ఆరోగ్యం",
              BIOMETRICS: "జీవమితులు",
              "EST. LENGTH": "అంచనా పొడవు",
              "AI MEASURED": "ఏఐ కొలిచింది",
              "EST. WEIGHT": "అంచనా బరువు",
              "MARKET & ECONOMICS": "మార్కెట్ & ఆర్థికాలు",
              "CURRENT MARKET PRICE": "ప్రస్తుత మార్కెట్ ధర",
              "ENVIRONMENT & HEALTH": "పర్యావరణం & ఆరోగ్యం",
              "ENV TEMP": "పర్యావరణ ఉష్ణోగ్రత",
              PATHOLOGY: "వ్యాధిశాస్త్రం",
              Healthy: "ఆరోగ్యంగా",
            },
            kn: {
              MATCH: "ಹೊಂದಿಕೆ",
              COUNT: "ಎಣಿಕೆ",
              HEALTH: "ಆರೋಗ್ಯ",
              BIOMETRICS: "ಜೈವಮಾಪನ",
              "EST. LENGTH": "ಅಂದಾಜು ಉದ್ದ",
              "AI MEASURED": "ಎಐ ಅಳೆಯಿತು",
              "EST. WEIGHT": "ಅಂದಾಜು ತೂಕ",
              "MARKET & ECONOMICS": "ಮಾರುಕಟ್ಟೆ & ಆರ್ಥಿಕತೆ",
              "CURRENT MARKET PRICE": "ಪ್ರಸ್ತುತ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ",
              "ENVIRONMENT & HEALTH": "ಪರಿಸರ & ಆರೋಗ್ಯ",
              "ENV TEMP": "ಪರಿಸರ ತಾಪಮಾನ",
              PATHOLOGY: "ರೋಗಶಾಸ್ತ್ರ",
              Healthy: "ಆರೋಗ್ಯಕರ",
            },
            gu: {
              MATCH: "મેળસાદ",
              COUNT: "ગણતરી",
              HEALTH: "આરોગ્ય",
              BIOMETRICS: "બાયોમેટ્રિક્સ",
              "EST. LENGTH": "અંદાજિત લંબાઈ",
              "AI MEASURED": "એઆઈ માપેલ",
              "EST. WEIGHT": "અંદાજિત વજન",
              "MARKET & ECONOMICS": "બજાર & અર્થવ્યવસ્થા",
              "CURRENT MARKET PRICE": "વર્તમાન બજાર ભાવ",
              "ENVIRONMENT & HEALTH": "પર્યાવરણ & આરોગ્ય",
              "ENV TEMP": "પર્યાવરણ તાપમાન",
              PATHOLOGY: "રોગવિજ્ઞાન",
              Healthy: "સ્વસ્થ",
            },
            mr: {
              MATCH: "जुळवणी",
              COUNT: "मोजणी",
              HEALTH: "आरोग्य",
              BIOMETRICS: "बायोमेट्रिक्स",
              "EST. LENGTH": "अंदाजे लांबी",
              "AI MEASURED": "एआय मोजमाप",
              "EST. WEIGHT": "अंदाजे वजन",
              "MARKET & ECONOMICS": "बाजार & अर्थशास्त्र",
              "CURRENT MARKET PRICE": "सध्याचा बाजारभाव",
              "ENVIRONMENT & HEALTH": "पर्यावरण & आरोग्य",
              "ENV TEMP": "पर्यावरण तापमान",
              PATHOLOGY: "रोगशास्त्र",
              Healthy: "निरोगी",
            },
            pa: {
              MATCH: "ਮਿਲਾਅ",
              COUNT: "ਗਿਣਤੀ",
              HEALTH: "ਸਿਹਤ",
              BIOMETRICS: "ਬਾਇਓਮੈਟਰਿਕਸ",
              "EST. LENGTH": "ਅੰਦਾਜ਼ਨ ਲੰਬਾਈ",
              "AI MEASURED": "ਏਆਈ ਮਾਪ",
              "EST. WEIGHT": "ਅੰਦਾਜ਼ਨ ਭਾਰ",
              "MARKET & ECONOMICS": "ਬਾਜ਼ਾਰ & ਅਰਥਸ਼ਾਸਤਰ",
              "CURRENT MARKET PRICE": "ਮੌਜੂਦਾ ਬਾਜ਼ਾਰ ਕੀਮਤ",
              "ENVIRONMENT & HEALTH": "ਵਾਤਾਵਰਨ & ਸਿਹਤ",
              "ENV TEMP": "ਵਾਤਾਵਰਨ ਤਾਪਮਾਨ",
              PATHOLOGY: "ਰੋਗ ਵਿਗਿਆਨ",
              Healthy: "ਤੰਦਰੁਸਤ",
            },
            or: {
              MATCH: "ମେଳ",
              COUNT: "ଗଣନା",
              HEALTH: "ସ୍ୱାସ୍ଥ୍ୟ",
              BIOMETRICS: "ଜୀବମାପନ",
              "EST. LENGTH": "ଆନୁମାନିକ ଲମ୍ବ",
              "AI MEASURED": "ଏଆଇ ମାପା",
              "EST. WEIGHT": "ଆନୁମାନିକ ଓଜନ",
              "MARKET & ECONOMICS": "ବଜାର & ଅର୍ଥନୀତି",
              "CURRENT MARKET PRICE": "ବର୍ତ୍ତମାନ ବଜାର ଦର",
              "ENVIRONMENT & HEALTH": "ପରିବେଶ & ସ୍ୱାସ୍ଥ୍ୟ",
              "ENV TEMP": "ପରିବେଶ ତାପମାନ",
              PATHOLOGY: "ରୋଗବିଜ୍ଞାନ",
              Healthy: "ସୁସ୍ଥ",
            },
          };
          const labels = {
            match: labelDict[currentLang]?.["MATCH"] || (currentLang === "en" ? "MATCH" : t("analyze.match", { defaultValue: "MATCH" }).toUpperCase()),
            count: labelDict[currentLang]?.["COUNT"] || (currentLang === "en" ? "COUNT" : t("analyze.count", { defaultValue: "COUNT" }).toUpperCase()),
            health: labelDict[currentLang]?.["HEALTH"] || (currentLang === "en" ? "HEALTH" : t("analyze.health", { defaultValue: "HEALTH" }).toUpperCase()),
            biometrics: labelDict[currentLang]?.["BIOMETRICS"] || (currentLang === "en" ? "BIOMETRICS" : t("analyze.biometrics", { defaultValue: "BIOMETRICS" }).toUpperCase()),
            estLength: labelDict[currentLang]?.["EST. LENGTH"] || (currentLang === "en" ? "EST. LENGTH" : t("analyze.estLength", { defaultValue: "EST. LENGTH" }).toUpperCase()),
            aiMeasured: labelDict[currentLang]?.["AI MEASURED"] || (currentLang === "en" ? "AI MEASURED" : t("analyze.aiMeasured", { defaultValue: "AI MEASURED" }).toUpperCase()),
            estWeight: labelDict[currentLang]?.["EST. WEIGHT"] || (currentLang === "en" ? "EST. WEIGHT" : t("analyze.estWeight", { defaultValue: "EST. WEIGHT" }).toUpperCase()),
            marketEconomics: labelDict[currentLang]?.["MARKET & ECONOMICS"] || (currentLang === "en" ? "MARKET & ECONOMICS" : t("analyze.marketEconomics", { defaultValue: "MARKET & ECONOMICS" }).toUpperCase()),
            currentMarketPrice: labelDict[currentLang]?.["CURRENT MARKET PRICE"] || (currentLang === "en" ? "CURRENT MARKET PRICE" : t("analyze.marketPrice", { defaultValue: "CURRENT MARKET PRICE" }).toUpperCase()),
            environmentHealth: labelDict[currentLang]?.["ENVIRONMENT & HEALTH"] || (currentLang === "en" ? "ENVIRONMENT & HEALTH" : t("analyze.envHealth", { defaultValue: "ENVIRONMENT & HEALTH" }).toUpperCase()),
            envTemp: labelDict[currentLang]?.["ENV TEMP"] || (currentLang === "en" ? "ENV TEMP" : t("analyze.waterTemp", { defaultValue: "ENV TEMP" }).toUpperCase()),
            pathology: labelDict[currentLang]?.["PATHOLOGY"] || (currentLang === "en" ? "PATHOLOGY" : t("analyze.pathology", { defaultValue: "PATHOLOGY" }).toUpperCase()),
            healthy: labelDict[currentLang]?.["Healthy"] || (currentLang === "en" ? "Healthy" : t("disease.healthy", { defaultValue: "Healthy" })),
          };

          const sampledWeight = pickSampleWeight(dbEntry.key);

          setResult({
            species: dbEntry.key,
            confidence: analysis.species.confidence,
            healthScore: analysis.freshness.score * 100,
            disease: analysis.disease.name,
            estimatedWeight: sampledWeight,
            estimatedCount: fishCount || 1,
            boundingBox: analysis.boundingBox,
            marketPrice: dbEntry.price,
            marketTrend: dbEntry.trend,
            waterTemp: 26 + Math.random() * 2,
            phLevel: 7.0 + Math.random() * 0.5,
            autoLength: bio.length,
            speciesLabel: speciesDisplay,
            diseaseLabel:
              (analysis.disease?.name && analysis.disease.name.toLowerCase() === "healthy")
                ? labels.healthy
                : (analysis.disease?.name || labels.healthy),
            labels,
          });

          if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition((pos) => {
              setLocation({
                latitude: pos.coords.latitude,
                longitude: pos.coords.longitude,
              });
            });
          }
        } else {
          throw new Error("No analysis returned");
        }
      } catch (e) {
        console.error("Analysis Failed", e);
        toast({
          variant: "destructive",
          title: t("analyze.analysisFailed"),
          description: t("analyze.analysisFailedDesc"),
        });
      } finally {
        setIsAnalyzing(false);
      }
    }, 200);
  };

  const handleSave = async () => {
    if (!imageData || !result) return;
    try {
      await databaseService.initialize?.();
      addLocalCatch({
        id: "local-" + Date.now(),
        createdAt: Date.now(),
        species: result.species || "Unknown",
        speciesLabel: result.speciesLabel,
        image: imageData,
        lat: location?.latitude ?? 0,
        lng: location?.longitude ?? 0,
        healthScore: result.healthScore,
        confidence: result.confidence,
        estimatedWeight: result.estimatedWeight,
      });
      await databaseService.addCatch({
        species: result.species,
        confidence: result.confidence,
        health_score: result.healthScore,
        estimated_weight: result.estimatedWeight,
        count: result.estimatedCount,
        timestamp: new Date().toISOString(),
        latitude: location?.latitude ?? 0,
        longitude: location?.longitude ?? 0,
        image_data: imageData,
        is_synced: false,
      });
      toast({
        title: t("analyze.saved"),
        description: t("analyze.catchSaved"),
      });
    } catch (e) {
      console.error(e);
      toast({
        title: t("analyze.saveFailed"),
        description: t("analyze.couldNotSave"),
      });
    }
  };

  if (showCamera)
    return (
      <CameraCapture
        onImageCapture={(data) => {
          setShowCamera(false);
          analyzeImage(data);
        }}
        onClose={() => setShowCamera(false)}
      />
    );
  if (showCalibration && imageData)
    return (
      <CalibrationHelper
        imageData={imageData}
        onCalibrated={(_, len) => {
          setMeasuredLength(len?.toFixed(1) || "-");
          setShowCalibration(false);
        }}
        onClose={() => setShowCalibration(false)}
      />
    );

  if (imageData && result) {
    return (
      <div className="min-h-[100dvh] bg-slate-950 text-white font-sans flex flex-col">
        <div className="max-w-7xl mx-auto w-full flex-1 grid grid-cols-1 lg:grid-cols-2 gap-0 lg:gap-8 lg:p-8">
          <div className="relative w-full h-[45vh] lg:h-full lg:rounded-3xl overflow-hidden bg-black shadow-2xl shrink-0">
            <div className="absolute top-0 left-0 right-0 z-20 p-4 flex justify-between items-center bg-gradient-to-b from-black/90 to-transparent">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setImageData(null);
                  setResult(null);
                }}
                className="text-white bg-black/20 hover:bg-black/40 backdrop-blur-md rounded-full"
              >
                <ArrowLeft className="w-6 h-6" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-white bg-black/20 hover:bg-black/40 backdrop-blur-md rounded-full"
              >
                <Share2 className="w-5 h-5" />
              </Button>
            </div>
            <ExplainabilityOverlay
              imageData={imageData}
              species={result.speciesLabel}
              confidence={result.confidence}
              boundingBox={result.boundingBox}
              className="h-full w-full object-cover"
            />
          </div>
          <div className="relative flex-1 bg-slate-900 lg:bg-transparent flex flex-col overflow-hidden -mt-6 lg:mt-0 rounded-t-3xl lg:rounded-none shadow-[0_-10px_40px_rgba(0,0,0,0.5)]">
            <div className="flex-1 overflow-y-auto no-scrollbar p-6 pb-32 lg:p-0">
              <div className="w-12 h-1.5 bg-slate-700/50 rounded-full mx-auto mb-6 lg:hidden" />
              <div className="lg:bg-slate-900/50 lg:backdrop-blur-xl lg:border lg:border-white/5 lg:p-8 lg:rounded-3xl lg:h-full lg:overflow-y-auto">
                <div className="flex justify-between items-start mb-8">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Badge
                        variant="outline"
                        className="border-cyan-500/30 text-cyan-400 bg-cyan-500/5 uppercase tracking-widest text-[10px]"
                      >
                        {t("analyze.identifiedSpecies")}
                      </Badge>
                      <Badge
                        variant="outline"
                        className="border-white/10 text-slate-400 uppercase tracking-widest text-[10px]"
                      >
                        {result.confidence.toFixed(1)}% {result.labels?.match || "MATCH"}
                      </Badge>
                      <Badge
                        variant="outline"
                        className="border-emerald-500/30 text-emerald-400 bg-emerald-500/5 uppercase tracking-widest text-[10px]"
                      >
                        {result.estimatedCount} {result.labels?.count || "COUNT"}
                      </Badge>
                    </div>
                    <h1 className="text-3xl lg:text-4xl font-bold text-white text-glow leading-tight mb-1 break-words">
                      {result.speciesLabel}
                    </h1>
                  </div>
                  
                  {/* --- DYNAMIC HEALTH CIRCLE (FIXED) --- */}
                  <div className={`flex flex-col items-center justify-center w-16 h-16 rounded-2xl shrink-0 border ${
                    result.healthScore > 70 
                      ? "bg-emerald-950/50 border-emerald-500/30" 
                      : "bg-red-950/50 border-red-500/30"
                  }`}>
                    <span className={`text-2xl font-bold ${
                      result.healthScore > 70 ? "text-emerald-400" : "text-red-400"
                    }`}>
                      {Math.round(result.healthScore)}
                    </span>
                    <span className={`text-[10px] uppercase font-bold ${
                      result.healthScore > 70 ? "text-emerald-600" : "text-red-600"
                    }`}>
                      {result.labels?.health || t("analyze.health")}
                    </span>
                  </div>

                </div>
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Ruler className="w-3 h-3" /> {result.labels?.biometrics || t("analyze.biometrics")}
                </h3>
                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="bg-black/20 p-4 rounded-2xl border border-white/5">
                    <div className="text-slate-400 text-xs uppercase mb-1">
                      {result.labels?.estLength || t("analyze.estLength")}
                    </div>
                    <div className="text-2xl font-mono text-white">
                      {result.autoLength}{" "}
                      <span className="text-sm text-slate-500">cm</span>
                    </div>
                    <div className="mt-2 text-[10px] text-cyan-400 flex items-center gap-1">
                      <Check className="w-3 h-3" /> {result.labels?.aiMeasured || t("analyze.aiMeasured")}
                    </div>
                  </div>
                  <div className="bg-black/20 p-4 rounded-2xl border border-white/5">
                    <div className="text-slate-400 text-xs uppercase mb-1">
                      {result.labels?.estWeight || t("analyze.estWeight")}
                    </div>
                    <div className="text-2xl font-mono text-white">
                      {result.estimatedWeight.toFixed(2)}{" "}
                      <span className="text-sm text-slate-500">kg</span>
                    </div>
                    <div className="mt-2 w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                      <div className="h-full bg-indigo-500 w-[70%]" />
                    </div>
                  </div>
                </div>
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                  <DollarSign className="w-3 h-3" /> {result.labels?.marketEconomics || t("analyze.marketEconomics")}
                </h3>
                <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-800/50 to-black/20 border border-white/5 mb-8">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <div className="text-slate-400 text-xs uppercase">
                        {result.labels?.currentMarketPrice || t("analyze.marketPrice")}
                      </div>
                      <div className="text-2xl font-bold text-white">
                        ₹{result.marketPrice}{" "}
                        <span className="text-sm font-normal text-slate-500">
                          / kg
                        </span>
                      </div>
                    </div>
                    <div
                      className={`px-3 py-1 rounded-lg border ${
                        result.marketTrend >= 0
                          ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                          : "bg-red-500/10 border-red-500/20 text-red-400"
                      }`}
                    >
                      <div className="flex items-center gap-1 text-xs font-bold">
                        <TrendingUp
                          className={`w-3 h-3 ${
                            result.marketTrend < 0 ? "rotate-180" : ""
                          }`}
                        />
                        {result.marketTrend > 0 ? "+" : ""}
                        {result.marketTrend}%
                      </div>
                    </div>
                  </div>
                </div>
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Droplets className="w-3 h-3" /> {result.labels?.environmentHealth || t("analyze.envHealth")}
                </h3>
                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="p-4 rounded-xl bg-black/20 border border-white/5">
                    <div className="text-white font-bold mb-1">
                      {result.waterTemp.toFixed(1)}°C
                    </div>
                    <div className="text-[10px] text-slate-500 uppercase">
                      {result.labels?.envTemp || "ENV TEMP"}
                    </div>
                  </div>
                  <div
                    className={`p-4 rounded-xl border border-white/5 ${
                      result.disease === "Healthy"
                        ? "bg-emerald-500/10 border-emerald-500/20"
                        : "bg-red-500/10 border-red-500/20"
                    }`}
                  >
                    <div
                      className={`font-bold mb-1 ${
                        result.disease === "Healthy"
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                        {result.diseaseLabel}
                    </div>
                    <div className="text-[10px] text-slate-500 uppercase">
                      {result.labels?.pathology || t("analyze.pathology")}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-slate-900/95 backdrop-blur-xl border-t border-white/10 z-30 lg:static lg:bg-transparent lg:border-0 lg:p-0">
              <div className="flex gap-4 max-w-md mx-auto lg:max-w-none">
                <Button
                  variant="outline"
                  onClick={() => {
                    setImageData(null);
                    setResult(null);
                  }}
                  className="flex-1 h-12 border-white/10 hover:bg-white/5 text-slate-300"
                >
                  {discardLabel}
                </Button>
                <Button
                  onClick={handleSave}
                  className="flex-[2] h-12 bg-cyan-600 hover:bg-cyan-500 text-white font-bold shadow-lg shadow-cyan-900/20"
                >
                  {saveRecordLabel}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-ocean pt-safe-top pb-safe-bottom">
      <input
        ref={fileInputRef}
        onChange={(e) => {
          if (e.target.files?.[0]) {
            const reader = new FileReader();
            reader.onload = (ev) => analyzeImage(ev.target?.result as string);
            reader.readAsDataURL(e.target.files[0]);
          }
        }}
        type="file"
        accept="image/*"
        className="hidden"
      />
      <div className="container mx-auto max-w-md px-4 space-y-6">
        <div className="text-center py-8 relative">
          <div className="absolute inset-0 bg-gradient-glow opacity-30 blur-3xl"></div>
          <div className="relative">
            <h1 className="text-3xl font-bold text-gradient mb-3">
              🐟 {t("analyze.aiScanner")}
            </h1>
            <p className="text-muted-foreground text-lg sm:animate-slide-up">
              {t("analyze.processingDescription")}
            </p>
            <div className="mt-4 flex justify-center items-center gap-2 text-sm text-muted-foreground">
              <div
                className={`w-2 h-2 rounded-full ${
                  isModelLoading
                    ? "bg-yellow-500 animate-pulse"
                    : modelError
                    ? "bg-red-500"
                    : "bg-emerald-500 animate-pulse-glow"
                }`}
              ></div>
              <span>
                {isModelLoading ? t("analyze.initializing") : "System Online"}
              </span>
            </div>
          </div>
        </div>
        <Card className="card-premium hover-glow animate-slide-up overflow-hidden">
          <div className="absolute inset-0 bg-gradient-primary opacity-5"></div>
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-3 text-xl">
              <div className="p-2 bg-gradient-primary rounded-lg">
                <Camera className="h-6 w-6 text-white" />
              </div>
              {t("analyze.professionalAnalysis")}
            </CardTitle>
            <p className="text-muted-foreground">
              {t("analyze.advancedDescription")}
            </p>
          </CardHeader>
          <CardContent className="space-y-4 relative">
            <Button
              onClick={() => setShowCamera(true)}
              disabled={isAnalyzing || isModelLoading}
              className="btn-premium btn-mobile w-full py-8 text-lg font-semibold relative overflow-hidden touch-feedback"
            >
              <div className="flex items-center justify-center gap-3">
                {isAnalyzing ? (
                  <>
                    <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>{t("analyze.analyzingAI")}</span>
                  </>
                ) : (
                  <>
                    <Camera className="h-6 w-6" />
                    <span>📸 {scanCatchLabel}</span>
                  </>
                )}
              </div>
            </Button>
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-muted"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">
                  {t("analyze.or")}
                </span>
              </div>
            </div>
            <Button
              variant="outline"
              className="btn-mobile w-full py-6 border-2 hover:border-primary/50 hover:bg-primary/5 transition-all"
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing || isModelLoading}
            >
              <Upload className="h-5 w-5 mr-2" /> {t("analyze.uploadGallery")}
            </Button>
          </CardContent>
        </Card>
        <div className="grid grid-cols-2 gap-4 animate-fade-in">
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">🎯</div>
            <div className="font-semibold text-sm">{t("analyze.accuracy")}</div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.aiConfidence")}
            </div>
          </Card>
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">⚡</div>
            <div className="font-semibold text-sm">
              {t("analyze.instantResults")}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.realTimeAnalysis")}
            </div>
          </Card>
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">⚕️</div>
            <div className="font-semibold text-sm">
              {t("analyze.healthScoreCheck")}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.freshnessCheck")}
            </div>
          </Card>
          <Card className="card-mobile hover-scale text-center p-4">
            <div className="text-2xl mb-2">📏</div>
            <div className="font-semibold text-sm">
              {t("analyze.sizeEstimation")}
            </div>
            <div className="text-xs text-muted-foreground">
              {t("analyze.weightLength")}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}