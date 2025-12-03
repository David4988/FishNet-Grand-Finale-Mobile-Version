import { useEffect, useMemo, useState, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { TFunction } from "i18next";
import { loadLocalCatches } from "@/utils/localCatches";
import { translateSpecies } from "@/utils/translateSpecies";
import { History, Filter, Zap, X, Droplet, BarChart } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";

// --- DUMMY IMPLEMENTATIONS FOR SELF-CONTAINED COMPONENT ---

type ButtonVariant = "ghost" | "default" | "outline" | string;
type ButtonSize = "icon" | "sm" | "default" | string;

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  className?: string;
  children: ReactNode;
}

const Button = ({ className, children, ...props }: ButtonProps) => (
  <button
    className={cn(
      "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background",
      className
    )}
    {...props}
  >
    {children}
  </button>
);

const Dialog = ({
  open,
  onOpenChange,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}) => (
  <AnimatePresence>
    {open && (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        onClick={() => onOpenChange(false)}
      >
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm" />
        {children}
      </motion.div>
    )}
  </AnimatePresence>
);

const DialogContent = ({
  className,
  children,
  ...props
}: {
  className?: string;
  children: ReactNode;
}) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95, y: 20 }}
    animate={{ opacity: 1, scale: 1, y: 0 }}
    exit={{ opacity: 0, scale: 0.95, y: 20 }}
    transition={{ duration: 0.3, ease: "easeOut" }}
    className={cn("relative", className)}
    onClick={(e) => e.stopPropagation()}
    {...props}
  >
    {children}
  </motion.div>
);

const DialogHeader = ({ children }: { children: ReactNode }) => (
  <div className="p-6 pb-0">{children}</div>
);
const DialogTitle = ({ children }: { children: ReactNode }) => (
  <h2 className="text-lg font-semibold">{children}</h2>
);

interface FishCatch {
  id: number | string;
  species: string;
  confidence: number;
  latitude: number;
  longitude: number;
  timestamp: string;
  image_data: string;
  health_score: number;
  estimated_weight: number;
}

const dummyCatches: FishCatch[] = [];

const databaseService = {
  getCatchesBySpecies: async (species: string): Promise<FishCatch[]> => {
    return Array.from({ length: 7 }).map((_, i) => ({
      ...(dummyCatches.find((c) => c.species === species) || dummyCatches[0])!,
      id: `trend-${i}`,
      timestamp: new Date(Date.now() - (7 - i) * 86400000).toISOString(),
      confidence: 85 + Math.random() * 10,
      health_score: 88 + Math.random() * 8,
    }));
  },
  getAllCatches: async (): Promise<FishCatch[]> => {
    return dummyCatches;
  },
};

const HistoryList = ({
  onCatchSelect,
  items,
  t,
}: {
  onCatchSelect: (c: FishCatch) => void;
  items: FishCatch[];
  t: TFunction;
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <motion.div
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      initial="hidden"
      animate="visible"
      variants={{ visible: { transition: { staggerChildren: 0.05 } } }}
    >
      {items.map((c, index) => (
        <motion.div
          key={c.id}
          onClick={() => onCatchSelect(c)}
          className="bg-slate-800/50 rounded-xl border border-sky-400/20 overflow-hidden cursor-pointer group hover:border-sky-400/50 transition-all relative"
          variants={{
            hidden: { opacity: 0, y: 20 },
            visible: { opacity: 1, y: 0 },
          }}
          whileHover={{ scale: 1.03 }}
          onMouseEnter={() => setHoveredIndex(index)}
          onMouseLeave={() => setHoveredIndex(null)}
        >
          <img
            src={c.image_data}
            alt={c.species}
            className="w-full h-48 object-cover"
          />
          <div className="p-4">
            <p className="font-bold text-lg text-white group-hover:text-sky-400 transition-colors">
              {translateSpecies(c.species, t)}
            </p>
            <p className="text-sm text-gray-400">
              {new Date(c.timestamp).toLocaleDateString()}
            </p>
            <div className="flex items-center gap-4 mt-3 text-xs text-gray-300">
              <span className="flex items-center gap-1.5">
                <Zap size={14} className="text-sky-400" />{" "}
                {c.confidence.toFixed(0)}% {t('analyze.confidence')}
              </span>
              <span className="flex items-center gap-1.5">
                <Droplet size={14} className="text-emerald-400" />{" "}
                {c.health_score.toFixed(0)}% {t('history.healthScore', { defaultValue: 'Health' })}
              </span>
            </div>
          </div>

          <AnimatePresence>
            {hoveredIndex === index && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="absolute inset-0 bg-slate-900/85 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center p-4 text-center"
              >
                <div className="space-y-3">
                  <p className="font-bold text-lg text-sky-400">
                    {translateSpecies(c.species, t)}
                  </p>
                  <p className="text-sm text-gray-300">
                    {new Date(c.timestamp).toLocaleDateString()}
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-center gap-1.5 text-xs text-sky-300">
                      <Zap size={12} className="text-sky-400" />
                      <span className="font-semibold">{c.confidence.toFixed(0)}%</span>
                      <span className="text-gray-400">{t('analyze.confidence')}</span>
                    </div>
                    <div className="flex items-center justify-center gap-1.5 text-xs text-emerald-300">
                      <Droplet size={12} className="text-emerald-400" />
                      <span className="font-semibold">{c.health_score.toFixed(0)}%</span>
                      <span className="text-gray-400">{t('history.healthScore', { defaultValue: 'Health' })}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      ))}
    </motion.div>
  );
};

const AnalysisResults = ({ result, t }: { result: FishCatch; t: TFunction }) => (
  <div className="text-gray-300 text-sm">
    {t('history.analysisDetails', { 
      species: translateSpecies(result.species, t),
      defaultValue: `Detailed analysis results for ${translateSpecies(result.species, t)} would be displayed here, providing deeper insights into the catch data and historical trends for regulatory and personal tracking purposes.`
    })}
  </div>
);

const cn = (...classes: (string | undefined | null | false)[]) =>
  classes.filter(Boolean).join(" ");

// --- END OF DUMMY IMPLEMENTATIONS ---

export default function HistoryPage() {
  const { t } = useTranslation();
  const [selected, setSelected] = useState<FishCatch | null>(null);
  const [catches, setCatches] = useState<FishCatch[]>([]);
  const [speciesHistory, setSpeciesHistory] = useState<FishCatch[]>([]);

  useEffect(() => {
    const loadAllData = async () => {
      // Load local catches from localStorage
      const localCatches = loadLocalCatches().map((lc) => ({
        id: lc.id,
        species: lc.speciesLabel || lc.species,
        confidence: lc.confidence ?? 0,
        latitude: lc.lat,
        longitude: lc.lng,
        timestamp: new Date(lc.createdAt).toISOString(),
        image_data: lc.image,
        health_score: lc.healthScore ?? 0,
        estimated_weight: lc.estimatedWeight ?? 0,
      }));

      // --- Combine database catches with public sample images ---
      const userCatches = await databaseService.getAllCatches();

      const publicFishImages = [
        "/fish/sole.jpg",
        "/fish/anchovy.jpg",
        "/fish/snapper.jpg",
        "/fish/tuna.jpg",
        "/fish/mackerel.jpg",
        "/fish/cod.jpg",
        "/fish/barracuda.jpg",
        "/fish/blackpomfret.jpg",
        "/fish/catfish.jpg",
        "/fish/crab.jpg",
        "/fish/squid.jpg",
      ];

      const filenameToSpecies = (path: string) => {
        const name =
          path
            .split("/")
            .pop()
            ?.replace(/\.(jpg|jpeg|png|webp)$/i, "") ?? "";
        return name
          .replace(/[_\-0-9]+/g, " ")
          .replace(/\s+/g, " ")
          .trim()
          .split(" ")
          .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
          .join(" ");
      };

      const sampleCatches = publicFishImages.map(
        (img, idx): FishCatch => ({
          id: `sample-${idx}`,
          species: filenameToSpecies(img),
          confidence: 90 + Math.floor(Math.random() * 10),
          latitude: 20.5937 + (Math.random() - 0.5) * 10,
          longitude: 78.9629 + (Math.random() - 0.5) * 10,
          timestamp: new Date(Date.now() - idx * 3 * 86400000).toISOString(),
          image_data: img,
          health_score: 85 + Math.floor(Math.random() * 15),
          estimated_weight: 1.2 + Math.random() * 2,
        })
      );

      // Prepend local catches, then user catches, then samples
      setCatches([...localCatches, ...userCatches, ...sampleCatches]);
    };
    loadAllData();
  }, []);

  useEffect(() => {
    const loadSpecies = async () => {
      if (selected) {
        // If the selected item is a sample, we generate fake history.
        // Otherwise, we would fetch real history.
        const list = await databaseService.getCatchesBySpecies(
          selected.species
        );
        setSpeciesHistory(list.slice(0, 12));
      } else {
        setSpeciesHistory([]);
      }
    };
    loadSpecies();
  }, [selected]);

  const chartData = useMemo(() => {
    return speciesHistory
      .slice()
      .reverse()
      .map((c) => ({
        label: new Date(c.timestamp).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
        }),
        Traceability: Math.round(c.confidence),
        Health: Math.round(c.health_score),
      }));
  }, [speciesHistory]);

  const chartConfig = {
    Traceability: { label: t('analyze.confidence'), color: "hsl(195, 89%, 52%)" },
    Health: { label: t('history.healthScore', { defaultValue: 'Health' }), color: "hsl(142, 71%, 45%)" },
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <div className="absolute inset-0 bg-grid-sky-400/[0.05]" />
      <style>{`
        :root {
          --color-Traceability: ${chartConfig.Traceability.color};
          --color-Health: ${chartConfig.Health.color};
        }
        .recharts-text.recharts-cartesian-axis-tick-value {
           fill: #9ca3af; 
           font-size: 12px;
        }
        .recharts-tooltip-wrapper {
            background-color: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(14, 165, 233, 0.3) !important;
            border-radius: 0.5rem !important;
            backdrop-filter: blur(4px);
        }
        .recharts-tooltip-label {
            color: white !important;
            font-weight: bold;
        }
      `}</style>

      <header className="sticky top-0 z-40 bg-slate-900/60 backdrop-blur-xl border-b border-sky-400/20 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <History className="h-6 w-6 text-sky-400" />
            <div>
              <h1 className="text-xl font-bold text-white">{t('history.title')}</h1>
              <p className="text-xs text-gray-400">{t('history.subtitle', { defaultValue: 'Your personal fishing log' })}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            className="text-gray-400 hover:text-white hover:bg-sky-500/20"
          >
            <Filter className="h-4 w-4 mr-2" />
            {t('common.filter')}
          </Button>
        </div>
      </header>

      <div className="container mx-auto max-w-7xl p-4">
        <HistoryList onCatchSelect={setSelected} items={catches} t={t} />
      </div>

      <Dialog open={!!selected} onOpenChange={(o) => !o && setSelected(null)}>
        <DialogContent className="max-w-2xl w-[90vw] max-h-[90vh] p-0 overflow-hidden bg-slate-900/80 backdrop-blur-2xl border border-sky-400/30 rounded-2xl shadow-2xl shadow-sky-400/10">
          {selected &&
            (() => {
              // Type guard for local catch
              function isLocalCatch(obj: any): obj is {
                image: string;
                createdAt: number;
                healthScore?: number;
                confidence?: number;
                lat: number;
                lng: number;
              } {
                return "image" in obj && "createdAt" in obj;
              }
              if (isLocalCatch(selected)) {
                const image = selected.image;
                const species = selected.species;
                const confidence = selected.confidence ?? 0;
                const health = selected.healthScore ?? 0;
                const date = new Date(selected.createdAt).toLocaleString();
                const lat = selected.lat;
                const lng = selected.lng;
                return (
                  <div
                    className="p-6 overflow-y-auto max-h-[90vh] space-y-6"
                    style={{
                      WebkitOverflowScrolling: "touch",
                      touchAction: "pan-y",
                    }}
                  >
                    <DialogHeader>
                      <DialogTitle>
                        <div className="flex justify-between items-start">
                          <span className="inline-flex items-center gap-3 text-2xl font-bold text-sky-400">
                            {species}
                          </span>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setSelected(null)}
                            className="text-gray-400 hover:text-white -mt-2 -mr-2"
                          >
                            <X />
                          </Button>
                        </div>
                      </DialogTitle>
                    </DialogHeader>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="flex flex-col items-center gap-4">
                        <img
                          src={image}
                          alt={species}
                          className="rounded-xl shadow-lg w-full object-cover border-2 border-sky-400/20"
                        />
                        <div className="grid grid-cols-2 gap-2 w-full text-center">
                          <div className="bg-sky-500/20 text-sky-300 px-3 py-2 rounded-lg text-sm font-semibold border border-sky-500/30">
                            <p className="text-xs text-sky-400">{t('analyze.confidence')}</p>
                            {confidence.toFixed(0)}%
                          </div>
                          <div className="bg-emerald-500/20 text-emerald-300 px-3 py-2 rounded-lg text-sm font-semibold border border-emerald-500/30">
                            <p className="text-xs text-emerald-400">
                              {t('history.healthScore', { defaultValue: 'Health Score' })}
                            </p>
                            {health.toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-slate-800/50 text-gray-300 px-3 py-2 rounded-lg text-sm font-medium w-full text-center border border-slate-700">
                          <div>{t('history.date')}: {date}</div>
                          <div>
                            {t('map.location')}: {lat}, {lng}
                          </div>
                        </div>
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold mb-2 text-white">
                          {t('analyze.results')}
                        </h3>
                        <div className="text-gray-300 text-sm">
                          <div>
                            <strong>{t('analyze.species')}:</strong> {species}
                          </div>
                          <div>
                            <strong>{t('history.date')}:</strong> {date}
                          </div>
                          <div>
                            <strong>{t('map.location')}:</strong> {lat}, {lng}
                          </div>
                          <div>
                            <strong>{t('analyze.confidence')}:</strong>{" "}
                            {confidence.toFixed(0)}%
                          </div>
                          <div>
                            <strong>{t('history.healthScore', { defaultValue: 'Health Score' })}:</strong> {health.toFixed(0)}%
                          </div>
                          <div>
                            <strong>Image:</strong>{" "}
                            <span className="break-all">{image}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              } else {
                // DB catch
                const image = selected.image_data;
                const species = selected.species;
                const confidence = selected.confidence;
                const health = selected.health_score;
                const date = new Date(selected.timestamp).toLocaleString();
                const lat = selected.latitude;
                const lng = selected.longitude;
                const weight = selected.estimated_weight;
                return (
                  <div className="p-6 overflow-y-auto max-h-[90vh] space-y-6">
                    <DialogHeader>
                      <DialogTitle>
                        <div className="flex justify-between items-start">
                          <span className="inline-flex items-center gap-3 text-2xl font-bold text-sky-400">
                            {species}
                          </span>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setSelected(null)}
                            className="text-gray-400 hover:text-white -mt-2 -mr-2"
                          >
                            <X />
                          </Button>
                        </div>
                      </DialogTitle>
                    </DialogHeader>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="flex flex-col items-center gap-4">
                        <img
                          src={image}
                          alt={species}
                          className="rounded-xl shadow-lg w-full object-cover border-2 border-sky-400/20"
                        />
                        <div className="grid grid-cols-2 gap-2 w-full text-center">
                          <div className="bg-sky-500/20 text-sky-300 px-3 py-2 rounded-lg text-sm font-semibold border border-sky-500/30">
                            <p className="text-xs text-sky-400">{t('analyze.confidence')}</p>
                            {confidence.toFixed(0)}%
                          </div>
                          <div className="bg-emerald-500/20 text-emerald-300 px-3 py-2 rounded-lg text-sm font-semibold border border-emerald-500/30">
                            <p className="text-xs text-emerald-400">
                              {t('history.healthScore', { defaultValue: 'Health Score' })}
                            </p>
                            {health.toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-slate-800/50 text-gray-300 px-3 py-2 rounded-lg text-sm font-medium w-full text-center border border-slate-700">
                          {t('history.weight')}: {weight.toFixed(1)} kg
                        </div>
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold mb-2 text-white">
                          {t('analyze.results')}
                        </h3>
                        <AnalysisResults result={selected} t={t} />
                      </div>
                    </div>

                    {chartData.length > 1 && (
                      <div>
                        <h3 className="text-lg font-semibold mb-2 text-white flex items-center gap-2">
                          <BarChart size={18} className="text-sky-400" /> {t('history.recentTrends', { defaultValue: 'Recent Trends' })}
                        </h3>
                        <div className="h-48 w-full -ml-4">
                          <ChartContainer
                            config={chartConfig}
                            className="h-full w-full"
                          >
                            <AreaChart
                              data={chartData}
                              margin={{
                                top: 10,
                                right: 20,
                                left: 10,
                                bottom: 0,
                              }}
                            >
                              <defs>
                                <linearGradient
                                  id="fillTraceability"
                                  x1="0"
                                  y1="0"
                                  x2="0"
                                  y2="1"
                                >
                                  <stop
                                    offset="5%"
                                    stopColor={chartConfig.Traceability.color}
                                    stopOpacity={0.8}
                                  />
                                  <stop
                                    offset="95%"
                                    stopColor={chartConfig.Traceability.color}
                                    stopOpacity={0.1}
                                  />
                                </linearGradient>
                                <linearGradient
                                  id="fillHealth"
                                  x1="0"
                                  y1="0"
                                  x2="0"
                                  y2="1"
                                >
                                  <stop
                                    offset="5%"
                                    stopColor={chartConfig.Health.color}
                                    stopOpacity={0.8}
                                  />
                                  <stop
                                    offset="95%"
                                    stopColor={chartConfig.Health.color}
                                    stopOpacity={0.1}
                                  />
                                </linearGradient>
                              </defs>
                              <CartesianGrid
                                vertical={false}
                                stroke="rgba(14, 165, 233, 0.15)"
                                strokeDasharray="3 3"
                              />
                              <XAxis
                                dataKey="label"
                                axisLine={false}
                                tickLine={false}
                              />
                              <YAxis
                                domain={[50, 100]}
                                axisLine={false}
                                tickLine={false}
                              />
                              <ChartTooltip
                                cursor={{
                                  stroke: "hsl(195, 89%, 52%)",
                                  strokeWidth: 1,
                                  fill: "transparent",
                                }}
                                content={
                                  <ChartTooltipContent indicator="line" />
                                }
                              />
                              <Area
                                type="monotone"
                                dataKey="Traceability"
                                strokeWidth={2}
                                stroke={chartConfig.Traceability.color}
                                fill="url(#fillTraceability)"
                              />
                              <Area
                                type="monotone"
                                dataKey="Health"
                                strokeWidth={2}
                                stroke={chartConfig.Health.color}
                                fill="url(#fillHealth)"
                              />
                            </AreaChart>
                          </ChartContainer>
                        </div>
                      </div>
                    )}
                  </div>
                );
              }
            })()}
        </DialogContent>
      </Dialog>
    </div>
  );
}
