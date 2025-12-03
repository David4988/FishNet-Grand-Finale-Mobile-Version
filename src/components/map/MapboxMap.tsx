import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FishCatch, databaseService } from "@/services/database";

interface MapboxMapProps {
  className?: string;
  onCatchSelect?: (catch_data: FishCatch) => void;
}

const INDIA_BOUNDS: [[number, number], [number, number]] = [[68.0, 6.5], [97.5, 37.0]];

export function MapboxMap({ className, onCatchSelect }: MapboxMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("MAPBOX_TOKEN"));
  const [catches, setCatches] = useState<FishCatch[]>([]);
  const [loc, setLoc] = useState<[number, number] | null>(null);

  // Load catches and user location
  useEffect(() => {
    databaseService
      .getAllCatches()
      .then((all) => {
        const valid = all.filter((c) => (c.latitude !== 0 || c.longitude !== 0));
        setCatches(valid);
      })
      .catch(console.warn);

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (p) => setLoc([p.coords.longitude, p.coords.latitude]),
        () => setLoc(null),
        { enableHighAccuracy: false, timeout: 5000 }
      );
    }
  }, []);

  // Initialize map when token available
  useEffect(() => {
    if (!token || !containerRef.current) return;

    mapboxgl.accessToken = token;
    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: "mapbox://styles/mapbox/light-v11",
      center: [78.9629, 20.5937],
      zoom: 4.5,
      pitch: 0,
      antialias: true,
    });
    mapRef.current = map;

    // Controls
    map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "top-right");

    map.on("load", () => {
      // Fit to India by default
      map.fitBounds(INDIA_BOUNDS, { padding: 40, duration: 0 });

      // If user location available, ease to it
      if (loc) {
        map.easeTo({ center: loc, zoom: 11, duration: 1000 });
        const el = document.createElement("div");
        el.style.width = "14px";
        el.style.height = "14px";
        el.style.borderRadius = "9999px";
        el.style.background = "#22c55e"; // semantic success color approx
        el.style.boxShadow = "0 0 0 4px rgba(34,197,94,0.25)";
        new mapboxgl.Marker({ element: el }).setLngLat(loc).addTo(map);
      }

      // Add catch markers
      catches.forEach((c) => {
        const el = document.createElement("div");
        el.style.width = "40px";
        el.style.height = "40px";
        el.style.borderRadius = "9999px";
        el.style.overflow = "hidden";
        el.style.border = `3px solid ${c.confidence >= 80 ? "#10b981" : c.confidence >= 60 ? "#f59e0b" : "#ef4444"}`;
        el.style.boxShadow = "0 6px 14px rgba(0,0,0,0.25)";

        const img = document.createElement("img");
        img.src = c.image_data;
        img.alt = c.species;
        img.style.width = "100%";
        img.style.height = "100%";
        img.style.objectFit = "cover";
        el.appendChild(img);

        if (onCatchSelect) {
          el.style.cursor = "pointer";
          el.onclick = () => onCatchSelect(c);
        }

        const marker = new mapboxgl.Marker({ element: el })
          .setLngLat([c.longitude, c.latitude])
          .setPopup(
            new mapboxgl.Popup({ offset: 12 }).setHTML(`
              <div style="min-width:200px">
                <div style="display:flex;gap:8px;align-items:center">
                  <img src="${c.image_data}" alt="${c.species}" style="width:48px;height:48px;border-radius:6px;object-fit:cover" />
                  <div>
                    <div style="font-weight:600">${c.species}</div>
                    <div style="font-size:12px;color:#6b7280">${(c.estimated_weight || 0).toFixed(1)} kg</div>
                  </div>
                </div>
                <div style="display:flex;gap:6px;margin-top:8px;font-size:12px">
                  <span>Conf: ${c.confidence.toFixed(0)}%</span>
                  <span>| Health: ${c.health_score.toFixed(0)}%</span>
                </div>
                <div style="font-size:11px;color:#6b7280;margin-top:4px">
                  ${new Date(c.timestamp).toLocaleString()}
                </div>
              </div>
            `)
          )
          .addTo(map);

        // Ensure popup loads image nicely
        marker.getElement().addEventListener("mouseenter", () => marker.togglePopup());
        marker.getElement().addEventListener("mouseleave", () => marker.togglePopup());
      });
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, containerRef, catches.length]);

  const [input, setInput] = useState("");
  const saveToken = () => {
    const t = input.trim();
    if (!t) return;
    localStorage.setItem("MAPBOX_TOKEN", t);
    setToken(t);
  };

  return (
    <div className={cn("relative h-screen", className)}>
      {!token && (
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <div className="bg-background/95 border shadow-lg rounded-lg p-4 max-w-sm w-full space-y-3">
            <div className="text-sm">Enter your Mapbox public token to enable the interactive map.</div>
            <input
              className="w-full px-3 py-2 rounded border bg-background"
              placeholder="pk.eyJ..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <div className="flex gap-2">
              <Button className="w-full" onClick={saveToken}>Save token</Button>
              <a
                className="text-primary text-sm underline"
                href="https://mapbox.com/"
                target="_blank" rel="noreferrer"
                aria-label="Get Mapbox token"
              >Get token</a>
            </div>
            <div className="text-xs text-muted-foreground">Tip: token is stored locally and can be changed anytime.</div>
          </div>
        </div>
      )}
      <div ref={containerRef} className="absolute inset-0 rounded-lg overflow-hidden" />
      <div className="absolute bottom-4 left-4 right-4 z-10">
        <div className="bg-background/90 backdrop-blur-sm rounded-lg p-3 shadow border">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{catches.length} catches mapped</span>
            {loc && <Badge variant="secondary">Near you</Badge>}
          </div>
        </div>
      </div>
    </div>
  );
}
