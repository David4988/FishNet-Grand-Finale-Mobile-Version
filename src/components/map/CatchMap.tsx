import { useEffect, useRef, useState } from "react";
import { loadLocalCatches } from "@/utils/localCatches";
import { motion, AnimatePresence } from "framer-motion";
// Import Leaflet from npm (you installed it with `npm install leaflet`)
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Award,
  Eye,
  Fish,
  Navigation,
  Users,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

// --- Dummy data and types to make the component self-contained ---
interface FishCatch {
  id: number | string;
  species: string;
  confidence: number;
  latitude: number;
  longitude: number;
  timestamp: string;
  image_data: string;
  health_score?: number;
  estimated_weight?: number;
  is_synced?: boolean;
}

const sampleIndiaFishCatches: FishCatch[] = [
  {
    id: "sample-1",
    species: "Indian Mackerel",
    confidence: 95,
    latitude: 13.0827,
    longitude: 80.2707,
    timestamp: new Date().toISOString(),
    image_data: "/fish/mackerel.jpg",
    health_score: 92,
    estimated_weight: 0.5,
  },
  {
    id: "sample-2",
    species: "Kingfish",
    confidence: 91,
    latitude: 15.2993,
    longitude: 74.124,
    timestamp: new Date().toISOString(),
    image_data: "/fish/kingfish.jpg",
    health_score: 88,
    estimated_weight: 2.1,
  },
  {
    id: "sample-3",
    species: "Pomfret",
    confidence: 88,
    latitude: 18.922,
    longitude: 72.8347,
    timestamp: new Date().toISOString(),
    image_data: "/fish/pomfret.jpg",
    health_score: 95,
    estimated_weight: 0.8,
  },
  {
    id: "sample-4",
    species: "Tuna",
    confidence: 98,
    latitude: 8.7642,
    longitude: 78.1348,
    timestamp: new Date().toISOString(),
    image_data: "/fish/tuna.jpg",
    health_score: 94,
    estimated_weight: 5.2,
  },
  // Add more sample data as needed
];

const databaseService = {
  getAllCatches: async (): Promise<FishCatch[]> => {
    return [
      {
        id: "user-1",
        species: "Red Snapper",
        confidence: 93,
        latitude: 13.0,
        longitude: 80.2,
        timestamp: new Date().toISOString(),
        image_data: "/fish/redsnapper.jpg",
        health_score: 91,
        estimated_weight: 1.5,
        is_synced: false,
      },
    ];
  },
};
// --- END OF DUMMY DATA ---

interface CatchMapProps {
  onCatchSelect?: (catch_data: FishCatch) => void;
  className?: string;
}

export const CatchMap = ({ onCatchSelect, className }: CatchMapProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const catchesLayerRef = useRef<L.LayerGroup | null>(null);
  const [sampleFish, setSampleFish] = useState<FishCatch[]>([]);
  const [catches, setCatches] = useState<FishCatch[]>([]);
  const [hoveredCatch, setHoveredCatch] = useState<FishCatch | null>(null);
  const [userLocation, setUserLocation] = useState<[number, number] | null>(
    null
  );
  const [locationError, setLocationError] = useState<string>("");
  const [selectedFilter, setSelectedFilter] = useState<
    "all" | "mine" | "community"
  >("all");
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnlineStatus = () => setIsOnline(navigator.onLine);
    window.addEventListener("online", handleOnlineStatus);
    window.addEventListener("offline", handleOnlineStatus);
    return () => {
      window.removeEventListener("online", handleOnlineStatus);
      window.removeEventListener("offline", handleOnlineStatus);
    };
  }, []);

  useEffect(() => {
    const loadData = async () => {
      // Load local catches from localStorage
      const localCatches = loadLocalCatches().map((lc) => ({
        id: lc.id,
        species: lc.species,
        confidence: lc.confidence ?? 0,
        latitude: lc.lat,
        longitude: lc.lng,
        timestamp: new Date(lc.createdAt).toISOString(),
        image_data: lc.image,
        health_score: lc.healthScore ?? 0,
        estimated_weight: 0,
        is_synced: false,
      }));

      const userCatches = await databaseService.getAllCatches();
      setCatches([...localCatches, ...userCatches]);
      setSampleFish(sampleIndiaFishCatches);
    };
    loadData();
  }, []);

  useEffect(() => {
    if (!navigator.geolocation) {
      setLocationError("Geolocation not supported");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserLocation([pos.coords.latitude, pos.coords.longitude]);
        setLocationError("");
      },
      () => setLocationError("Location unavailable")
    );
  }, []);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    // Ensure default marker icons are available when using the npm package
    // (Leaflet's images are not automatically discovered in some bundlers)
    const proto = L.Icon.Default.prototype as unknown as Record<
      string,
      unknown
    >;
    if (typeof proto._getIconUrl !== "undefined") {
      // attempt to remove any baked-in URL resolver
      delete proto._getIconUrl as unknown as void;
    }
    L.Icon.Default.mergeOptions({
      iconRetinaUrl:
        "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
      iconUrl:
        "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
      shadowUrl:
        "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
    });

    // Relax bounds to allow free panning/zooming; removing restrictive maxBounds prevents snapping
    const map = L.map(containerRef.current, {
      center: [15.0, 78.0],
      zoom: 5,
      zoomControl: false,
      attributionControl: false,
      // Allow panning beyond any bounds to keep the map interactive
      // Removing maxBounds and setting viscosity to 0 prevents auto-snap back
      maxBoundsViscosity: 0,
      minZoom: 4,
      worldCopyJump: true,
    });

    // use a light basemap for a light-themed UI
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
    }).addTo(map);

    mapRef.current = map;
    catchesLayerRef.current = L.layerGroup().addTo(map);
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Helpful debug flag for rendering a user-facing message when Leaflet isn't loaded
  const leafletAvailable = typeof L !== "undefined";

  useEffect(() => {
    if (!mapRef.current || !catchesLayerRef.current) return;
    const group = catchesLayerRef.current;
    group.clearLayers();

    let visible: FishCatch[] = [];
    if (selectedFilter === "all") visible = [...catches, ...sampleFish];
    else if (selectedFilter === "mine") visible = catches;
    else visible = sampleFish;

    visible.forEach((c) => {
      if (typeof c.latitude !== "number" || typeof c.longitude !== "number")
        return;

  const isUserCatch = catches.some((uc) => uc.id === c.id);
  const markerImage = c.image_data;

      const borderColor = isUserCatch ? "#f59e0b" : "#0ea5e9";
      const shadowColor = isUserCatch
        ? "rgba(245, 158, 11, 0.7)"
        : "rgba(14, 165, 233, 0.7)";

      const customIcon = L.divIcon({
        className: "custom-fish-marker",
        html: `<div class="fish-marker-icon" style="border-color: ${borderColor}; box-shadow: 0 0 15px ${shadowColor};"><img src="${markerImage}" alt="${c.species}" /></div>`,
        iconSize: [40, 40],
        iconAnchor: [20, 20],
      });

      const marker = L.marker([c.latitude, c.longitude], {
        title: c.species,
        icon: customIcon,
      });

      marker.on("mouseover", (e: L.LeafletMouseEvent) => {
        setHoveredCatch(c);
        // marker.getElement() is available on Marker instances; narrow the type safely
        const markerEl = (e.target as L.Marker).getElement?.();
        const icon = markerEl?.querySelector?.(
          ".fish-marker-icon"
        ) as HTMLElement | null;
        if (icon) {
          icon.style.transform = "scale(1.3)";
          icon.style.borderColor = "#67e8f9";
          icon.style.zIndex = "1000";
          icon.style.animation = "none";
        }
      });

      marker.on("mouseout", (e: L.LeafletMouseEvent) => {
        setHoveredCatch(null);
        const markerEl = (e.target as L.Marker).getElement?.();
        const icon = markerEl?.querySelector?.(
          ".fish-marker-icon"
        ) as HTMLElement | null;
        if (icon) {
          icon.style.transform = "scale(1)";
          icon.style.borderColor = borderColor;
          icon.style.zIndex = "1";
          icon.style.animation = isUserCatch
            ? "pulse-user 2s infinite"
            : "pulse 2s infinite";
        }
      });

      marker.on("click", () => onCatchSelect?.(c));
      marker.addTo(group);
    });
  }, [catches, sampleFish, selectedFilter, onCatchSelect]);

  const centerOnUserLocation = () => {
    if (userLocation && mapRef.current) {
      mapRef.current.setView(userLocation, 13);
    }
  };

  return (
    <div className={cn("relative h-full bg-white", className)}>
      <style>{`
        .fish-marker-icon {
          width: 40px; height: 40px; border-radius: 50%; overflow: hidden;
          border: 3px solid; background: #ffffff;
          cursor: pointer; transition: all 0.2s ease-in-out;
          animation: pulse 2s infinite;
        }
        .fish-marker-icon img { width: 100%; height: 100%; object-fit: cover; }
        @keyframes pulse {
          0%, 100% { box-shadow: 0 0 12px rgba(14, 165, 233, 0.18); }
          50% { box-shadow: 0 0 20px rgba(14, 165, 233, 0.24); }
        }
        /* user markers have a warmer pulse */
        .fish-marker-icon[style*="#f59e0b"] { animation-name: pulse-user; }
        @keyframes pulse-user {
          0%, 100% { box-shadow: 0 0 12px rgba(245, 158, 11, 0.18); }
          50% { box-shadow: 0 0 20px rgba(245, 158, 11, 0.26); }
        }
      `}</style>

      {/* Leaflet is imported from node_modules; CDN tags removed to avoid duplicate loads */}

      {/* make sure the container has an explicit min-height so h-full has something to stretch */}
      <div
        ref={containerRef}
        className="absolute inset-0 h-full rounded-xl overflow-hidden"
        style={{ minHeight: 360 }}
      />

      {/* If Leaflet isn't available, show a clear overlay with instructions to help debugging */}
      {!leafletAvailable && (
        <div className="absolute inset-0 z-[3000] flex items-center justify-center bg-white/95 text-slate-900 p-4 text-center border border-slate-200">
          <div>
            <div className="font-bold text-lg mb-2">
              Map not initialized — Leaflet missing
            </div>
            <div className="text-sm mb-3">
              The global <code>L</code> (Leaflet) object is not present. Please
              ensure Leaflet is loaded before this component mounts. e.g. add
              the following to your <code>index.html</code> inside{" "}
              <code>&lt;head&gt;</code>:
            </div>
            <pre className="bg-gray-50 p-3 rounded text-xs text-left border border-slate-100">
              {`<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />\n<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>`}
            </pre>
            <div className="mt-3 text-xs opacity-80">
              You can also install Leaflet as a dependency and import it
              (recommended for production).
            </div>
          </div>
        </div>
      )}
      <AnimatePresence>
        {hoveredCatch && (
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="absolute left-1/2 top-4 z-[2000] -translate-x-1/2 bg-white/90 backdrop-blur-sm rounded-xl shadow-2xl shadow-slate-300/30 border border-slate-200 p-4 w-80 max-w-full"
          >
            <div className="flex items-center gap-4 mb-3">
              <img
                src={hoveredCatch.image_data}
                alt="Fish"
                className="w-20 h-20 rounded-lg object-cover border-2 border-slate-200"
              />
              <div>
                <div className="font-bold text-lg text-slate-900">
                  {hoveredCatch.species}
                </div>
                <div className="text-xs text-slate-500 mb-2">
                  {new Date(hoveredCatch.timestamp).toLocaleString()}
                </div>
                {hoveredCatch.estimated_weight && (
                  <div className="text-xs text-slate-700 font-semibold">
                    Weight: {hoveredCatch.estimated_weight.toFixed(1)} kg
                  </div>
                )}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-sky-50 text-sky-700 border border-sky-100">
                <Zap className="w-3 h-3 text-sky-400" />
                <span className="font-bold">Traceability:</span>{" "}
                {hoveredCatch.confidence.toFixed(0)}%
              </span>
              {typeof hoveredCatch.health_score === "number" && (
                <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-100">
                  <span className="text-emerald-500">💚</span>
                  <span className="font-bold">Health:</span>{" "}
                  {hoveredCatch.health_score.toFixed(0)}%
                </span>
              )}
            </div>
            {hoveredCatch.latitude && hoveredCatch.longitude && (
              <div className="text-xs text-slate-500 mt-2">
                GPS: {hoveredCatch.latitude.toFixed(4)},{" "}
                {hoveredCatch.longitude.toFixed(4)}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
      <div className="absolute top-6 right-6 z-[1000] space-y-3">
        <Button
          size="icon"
          onClick={centerOnUserLocation}
          className="bg-white/90 backdrop-blur-sm hover:bg-sky-50 shadow-md border border-slate-200 text-slate-700 hover:text-slate-900"
          title="Center on my location"
        >
          <Navigation size={18} />
        </Button>
        <Card className="bg-white/90 backdrop-blur-sm shadow-xl border border-slate-200 text-slate-800">
          <CardContent className="p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-gray-400 mb-2">
              <Eye className="w-3 h-3" /> Map Layers
            </div>
            <div className="flex flex-col gap-1">
              {(() => {
                type FilterKey = "all" | "mine" | "community";
                interface FilterItem {
                  key: FilterKey;
                  label: string;
                  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
                  count: number;
                }
                const filters: FilterItem[] = [
                  {
                    key: "all",
                    label: "All Catches",
                    icon: Fish,
                    count: catches.length + sampleFish.length,
                  },
                  {
                    key: "mine",
                    label: "My Catches",
                    icon: Award,
                    count: catches.length,
                  },
                  {
                    key: "community",
                    label: "Community Data",
                    icon: Users,
                    count: sampleFish.length,
                  },
                ];

                return filters.map((filter) => {
                  const Icon = filter.icon;
                  return (
                    <Button
                      key={filter.key}
                      size="sm"
                      variant={
                        selectedFilter === filter.key ? "default" : "ghost"
                      }
                      onClick={() => setSelectedFilter(filter.key)}
                      className={cn(
                        "justify-between text-xs h-8 w-full",
                        selectedFilter === filter.key
                          ? "bg-sky-500 text-white"
                          : "hover:bg-sky-50 text-slate-700"
                      )}
                    >
                      <div className="flex items-center gap-2">
                        <Icon className="w-3 h-3" />
                        <span>{filter.label}</span>
                      </div>
                      <Badge
                        variant={
                          selectedFilter === filter.key
                            ? "default"
                            : "secondary"
                        }
                        className="ml-2 text-xs"
                      >
                        {filter.count}
                      </Badge>
                    </Button>
                  );
                });
              })()}
            </div>
          </CardContent>
        </Card>
      </div>
      <div className="absolute bottom-[15%] left-1/2 -translate-x-1/2 z-[1000] w-[calc(100%-3rem)] max-w-lg">
        <Card className="bg-white/90 backdrop-blur-sm shadow-2xl border border-slate-200">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-sky-500 rounded-lg shadow-lg shadow-sky-500/20">
                  <Fish size={20} className="text-white" />
                </div>
                <div>
                  <div className="font-bold text-lg text-slate-900">
                    {catches.length + sampleFish.length} Catches Mapped
                  </div>
                  <div className="text-sm text-slate-500">
                    {catches.length > 0 &&
                      `Avg. Traceability: ${(
                        catches.reduce((s, c) => s + c.confidence, 0) /
                        catches.length
                      ).toFixed(0)}%`}
                  </div>
                </div>
              </div>
              <div className="text-right space-y-1">
                <Badge
                  variant="outline"
                  className={cn(
                    "text-xs",
                    isOnline
                      ? "bg-emerald-50 text-emerald-700 border-emerald-100"
                      : "bg-amber-50 text-amber-700 border-amber-100"
                  )}
                >
                  {isOnline ? (
                    <>
                      <Wifi className="h-3 w-3 mr-1" /> Online
                    </>
                  ) : (
                    <>
                      <WifiOff className="h-3 w-3 mr-1" /> Offline
                    </>
                  )}
                </Badge>
                {catches.filter((c) => !c.is_synced).length > 0 && (
                  <Badge
                    variant="outline"
                    className="bg-amber-50 text-amber-700 border-amber-100 block"
                  >
                    {catches.filter((c) => !c.is_synced).length} Pending Sync
                  </Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
