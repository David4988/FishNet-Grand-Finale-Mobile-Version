import { useEffect, useMemo, useState, ReactNode } from "react";
import { History, Filter, Zap, X, Droplet, BarChart } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";

// --- DUMMY IMPLEMENTATIONS FOR SELF-CONTAINED COMPONENT ---

const Button = ({
  className,
  children,
  ...props
}: {
  className?: string;
  children: ReactNode;
  [key: string]: any;
}) => (
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

// --- FIX: Replaced Unsplash URLs with reliable placeholders ---
const dummyCatches: FishCatch[] = [
  {
    id: 1,
    species: "Indian Mackerel",
    confidence: 95,
    latitude: 13.08,
    longitude: 80.27,
    timestamp: new Date(Date.now() - 86400000).toISOString(),
    image_data: "https://placehold.co/400x300/0891b2/ffffff?text=Mackerel",
    health_score: 92,
    estimated_weight: 0.5,
  },
  {
    id: 2,
    species: "Red Snapper",
    confidence: 93,
    latitude: 13.0,
    longitude: 80.2,
    timestamp: new Date().toISOString(),
    image_data: "https://placehold.co/400x300/dc2626/ffffff?text=Snapper",
    health_score: 91,
    estimated_weight: 1.5,
  },
  {
    id: 3,
    species: "Kingfish",
    confidence: 91,
    latitude: 15.29,
    longitude: 74.12,
    timestamp: new Date(Date.now() - 2 * 86400000).toISOString(),
    image_data: "https://placehold.co/400x300/f59e0b/ffffff?text=Kingfish",
    health_score: 88,
    estimated_weight: 2.1,
  },
];

const databaseService = {
  getCatchesBySpecies: async (species: string): Promise<FishCatch[]> => {
    return Array.from({ length: 7 }).map((_, i) => ({
      ...dummyCatches.find((c) => c.species === species)!,
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

export const HistoryList = ({
  onCatchSelect,
  items,
}: {
  onCatchSelect: (c: FishCatch) => void;
  items: FishCatch[];
}) => (
  <motion.div
    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
    initial="hidden"
    animate="visible"
    variants={{ visible: { transition: { staggerChildren: 0.05 } } }}
  >
    {/* fallback images used when an item has no image_data */}
    {(() => {
      const fallback = [
        "/fish/anchovy.jpg",
        "/fish/barracuda.jpg",
        "/fish/mackerel.jpg",
        "/fish/tuna.jpg",
        "/fish/pomfret.jpg",
        "/fish/snapper.jpg",
      ];
      return items.map((c, idx) => {
        const imgSrc =
          c.image_data && c.image_data.length
            ? c.image_data
            : fallback[idx % fallback.length];
        return (
          <motion.div
            key={c.id}
            onClick={() => onCatchSelect(c)}
            className="bg-slate-800/50 rounded-xl border border-sky-400/20 overflow-hidden cursor-pointer group hover:border-sky-400/50 transition-all"
            variants={{
              hidden: { opacity: 0, y: 20 },
              visible: { opacity: 1, y: 0 },
            }}
            whileHover={{ scale: 1.03 }}
          >
            <img
              src={imgSrc}
              alt={c.species}
              className="w-full h-48 object-cover"
            />
            <div className="p-4">
              <p className="font-bold text-lg text-white group-hover:text-sky-400 transition-colors">
                {c.species}
              </p>
              <p className="text-sm text-gray-400">
                {new Date(c.timestamp).toLocaleDateString()}
              </p>
              <div className="flex items-center gap-4 mt-3 text-xs text-gray-300">
                <span className="flex items-center gap-1.5">
                  <Zap size={14} className="text-sky-400" />{" "}
                  {c.confidence.toFixed(0)}% Traceability
                </span>
                <span className="flex items-center gap-1.5">
                  <Droplet size={14} className="text-emerald-400" />{" "}
                  {c.health_score.toFixed(0)}% Health
                </span>
              </div>
            </div>
          </motion.div>
        );
      });
    })()}
  </motion.div>
);

const AnalysisResults = ({ result }: { result: any }) => (
  <div className="text-gray-300 text-sm">
    Detailed analysis results for {result.species} would be displayed here,
    providing deeper insights into the catch data and historical trends for
    regulatory and personal tracking purposes.
  </div>
);

const cn = (...classes: (string | undefined | null | false)[]) =>
  classes.filter(Boolean).join(" ");

// --- END OF DUMMY IMPLEMENTATIONS ---

export default function HistoryPage() {
  const [selected, setSelected] = useState<FishCatch | null>(null);
  const [catches, setCatches] = useState<FishCatch[]>([]);
  const [speciesHistory, setSpeciesHistory] = useState<FishCatch[]>([]);

  useEffect(() => {
    const loadAllCatches = async () => {
      const all = await databaseService.getAllCatches();
      setCatches(all);
    };
    loadAllCatches();
  }, []);

  useEffect(() => {
    const loadSpecies = async () => {
      if (selected) {
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
    Traceability: { label: "Traceability", color: "hsl(195, 89%, 52%)" }, // sky-500
    Health: { label: "Health", color: "hsl(142, 71%, 45%)" }, // emerald-500
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
              <h1 className="text-xl font-bold text-white">Catch History</h1>
              <p className="text-xs text-gray-400">Your personal fishing log</p>
            </div>
          </div>
          <Button
            variant="ghost"
            className="text-gray-400 hover:text-white hover:bg-sky-500/20"
          >
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
        </div>
      </header>

      <div className="container mx-auto max-w-7xl p-4">
        <HistoryList onCatchSelect={setSelected} items={catches} />
      </div>

      <Dialog open={!!selected} onOpenChange={(o) => !o && setSelected(null)}>
        <DialogContent className="max-w-2xl w-[90vw] max-h-[90vh] p-0 overflow-hidden bg-slate-900/80 backdrop-blur-2xl border border-sky-400/30 rounded-2xl shadow-2xl shadow-sky-400/10">
          {selected && (
            <div
              className="p-6 overflow-y-auto max-h-[90vh] space-y-6"
              style={{ WebkitOverflowScrolling: "touch", touchAction: "pan-y" }}
            >
              <DialogHeader>
                <DialogTitle>
                  <div className="flex justify-between items-start">
                    <span className="inline-flex items-center gap-3 text-2xl font-bold text-sky-400">
                      {selected.species}
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
                    src={selected.image_data}
                    alt={selected.species}
                    className="rounded-xl shadow-lg w-full object-cover border-2 border-sky-400/20"
                  />
                  <div className="grid grid-cols-2 gap-2 w-full text-center">
                    <div className="bg-sky-500/20 text-sky-300 px-3 py-2 rounded-lg text-sm font-semibold border border-sky-500/30">
                      <p className="text-xs text-sky-400">Traceability</p>
                      {selected.confidence.toFixed(0)}%
                    </div>
                    <div className="bg-emerald-500/20 text-emerald-300 px-3 py-2 rounded-lg text-sm font-semibold border border-emerald-500/30">
                      <p className="text-xs text-emerald-400">Health Score</p>
                      {selected.health_score.toFixed(0)}%
                    </div>
                  </div>
                  <div className="bg-slate-800/50 text-gray-300 px-3 py-2 rounded-lg text-sm font-medium w-full text-center border border-slate-700">
                    Weight: {selected.estimated_weight.toFixed(1)} kg
                  </div>
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2 text-white">
                    Analysis Details
                  </h3>
                  <AnalysisResults result={selected as any} />
                </div>
              </div>

              {chartData.length > 1 && (
                <div>
                  <h3 className="text-lg font-semibold mb-2 text-white flex items-center gap-2">
                    <BarChart size={18} className="text-sky-400" /> Recent
                    Trends
                  </h3>
                  <div className="h-48 w-full -ml-4">
                    <ChartContainer
                      config={chartConfig}
                      className="h-full w-full"
                    >
                      <AreaChart
                        data={chartData}
                        margin={{ top: 10, right: 20, left: 10, bottom: 0 }}
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
                          content={<ChartTooltipContent indicator="line" />}
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
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
