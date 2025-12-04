import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FishCatch, databaseService } from "@/services/database";

const INDIA_BOUNDS = new mapboxgl.LngLatBounds(
  [68.0, 6.5],
  [97.5, 37.0]
);

interface MapboxMapProps {
  className?: string;
  onCatchSelect?: (catch_data: FishCatch) => void;
}

export function MapboxMap({ className, onCatchSelect }: MapboxMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const markersRef = useRef<mapboxgl.Marker[]>([]);
  
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("MAPBOX_TOKEN"));
  const [catches, setCatches] = useState<FishCatch[]>([]);
  const [loc, setLoc] = useState<[number, number] | null>(null);
  const [input, setInput] = useState("");

  // Load catches
  useEffect(() => {
    databaseService
      .getAllCatches()
      .then((all) => {
        const valid = all.filter((c) => (c.latitude !== 0 || c.longitude !== 0));
        setCatches(valid);
      })
      .catch(console.warn);
  }, []);

  // Initialize Map
  useEffect(() => {
    if (!token || !containerRef.current || mapRef.current) return;

    try {
      mapboxgl.accessToken = token;
      const map = new mapboxgl.Map({
        container: containerRef.current,
        style: "mapbox://styles/mapbox/light-v11",
        center: [78.9629, 20.5937],
        zoom: 4,
        pitch: 0,
        attributionControl: false,
      });

      mapRef.current = map;

      // --- NEW: Add Standard Navigation Controls ---
      map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "top-right");

      // --- NEW: Add "Locate Me" Button (Fixes GPS issue) ---
      const geolocate = new mapboxgl.GeolocateControl({
        positionOptions: {
          enableHighAccuracy: true
        },
        trackUserLocation: true,
        showUserHeading: true
      });
      map.addControl(geolocate, "top-right");

      // Listen for when the user clicks the GPS button
      geolocate.on('geolocate', (e: any) => {
        const lon = e.coords.longitude;
        const lat = e.coords.latitude;
        setLoc([lon, lat]);
      });

      map.on("load", () => {
        map.fitBounds(INDIA_BOUNDS, { padding: 20, duration: 1000 });
      });

    } catch (error) {
      console.error("Mapbox init failed:", error);
    }

    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, [token]);

  // Handle Markers
  useEffect(() => {
    const map = mapRef.current;
    if (!map || catches.length === 0) return;

    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    catches.forEach((c) => {
      const el = document.createElement("div");
      el.className = "catch-marker";
      el.style.width = "40px";
      el.style.height = "40px";
      el.style.borderRadius = "50%";
      el.style.overflow = "hidden";
      el.style.border = `3px solid ${c.confidence >= 80 ? "#10b981" : c.confidence >= 60 ? "#f59e0b" : "#ef4444"}`;
      el.style.boxShadow = "0 6px 14px rgba(0,0,0,0.25)";
      el.style.backgroundColor = "white";

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

      const popupHTML = `
        <div style="min-width:180px; padding: 4px;">
          <div style="display:flex;gap:8px;align-items:center">
            <img src="${c.image_data}" style="width:40px;height:40px;border-radius:6px;object-fit:cover" />
            <div>
              <div style="font-weight:700;font-size:14px">${c.species}</div>
              <div style="font-size:12px;color:#6b7280">${(c.estimated_weight || 0).toFixed(1)} kg</div>
            </div>
          </div>
        </div>
      `;

      const marker = new mapboxgl.Marker({ element: el })
        .setLngLat([c.longitude, c.latitude])
        .setPopup(new mapboxgl.Popup({ offset: 25, closeButton: false }).setHTML(popupHTML))
        .addTo(map);

      el.addEventListener('mouseenter', () => marker.togglePopup());
      el.addEventListener('mouseleave', () => marker.togglePopup());

      markersRef.current.push(marker);
    });

  }, [catches, token]);

  const saveToken = () => {
    const t = input.trim();
    if (!t) return;
    localStorage.setItem("MAPBOX_TOKEN", t);
    setToken(t);
  };

  return (
    <div className={cn("relative w-full h-full", className)}>
      {!token && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/80 backdrop-blur-sm p-4">
          <div className="bg-card border shadow-xl rounded-xl p-6 max-w-sm w-full space-y-4">
            <h3 className="font-semibold text-lg">Map Configuration</h3>
            <p className="text-sm text-muted-foreground">Enter your Mapbox public token to view the catch map.</p>
            <input
              className="w-full px-3 py-2 rounded-md border bg-background"
              placeholder="pk.eyJ..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <div className="flex gap-2">
              <Button className="w-full" onClick={saveToken}>Save Token</Button>
            </div>
            <div className="text-xs text-center text-muted-foreground">
              <a href="https://mapbox.com/" target="_blank" className="underline hover:text-primary">Get a free token here</a>
            </div>
          </div>
        </div>
      )}
      
      {/* Map Container */}
      <div 
        ref={containerRef} 
        className="w-full h-full absolute inset-0 rounded-lg overflow-hidden" 
        style={{ minHeight: '400px' }} 
      />

      {/* Stats Overlay */}
      {token && (
        <div className="absolute bottom-6 left-4 right-4 z-10">
          <div className="bg-background/90 backdrop-blur-md rounded-xl p-4 shadow-lg border border-border/50 flex justify-between items-center">
            <div>
              <span className="font-bold text-lg">{catches.length}</span>
              <span className="text-sm text-muted-foreground ml-2">Catches Mapped</span>
            </div>
            {loc && <Badge className="bg-green-500/10 text-green-600 hover:bg-green-500/20 border-green-200">GPS Active</Badge>}
          </div>
        </div>
      )}
    </div>
  );
}