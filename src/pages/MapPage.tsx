import { useState, useEffect } from 'react';
import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { MapPin, Newspaper, Cloud, Menu } from 'lucide-react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { CatchMap } from '@/components/map/CatchMap';
import { NewsModule } from '@/components/map/NewsModule';
import { WeatherModule } from '@/components/map/WeatherModule';
import { cn } from '@/lib/utils';

type SectionType = 'map' | 'news' | 'weather';

interface NavigationItem {
  id: SectionType;
  icon: any;
  labelKey: string;
}

export default function MapPage() {
  const { t } = useTranslation();
  const [activeSection, setActiveSection] = useState<SectionType>('map');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const sectionLabel =
    activeSection === 'map'
      ? t('map.tabs.map', { defaultValue: 'Map' })
      : activeSection === 'news'
      ? t('map.tabs.news', { defaultValue: 'News' })
      : t('map.tabs.weather', { defaultValue: 'Weather' });

  const navItems: NavigationItem[] = [
    { id: 'map', icon: MapPin, labelKey: 'map.tabs.map' },
    { id: 'news', icon: Newspaper, labelKey: 'map.tabs.news' },
    { id: 'weather', icon: Cloud, labelKey: 'map.tabs.weather' },
  ];

  // Fix CatchMap incorrect size on visibility change
  useEffect(() => {
    if (activeSection === 'map') {
      const timer = setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [activeSection]);

  // Close menu on Escape and outside click for interactivity
  useEffect(() => {
    if (!isSidebarOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsSidebarOpen(false);
    };
    const onClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (menuRef.current && !menuRef.current.contains(target)) {
        setIsSidebarOpen(false);
      }
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onClickOutside);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onClickOutside);
    };
  }, [isSidebarOpen]);

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header fixed to always appear above the map on first load */}
      <header className="fixed top-0 left-0 right-0 z-[3000] bg-card/80 backdrop-blur-md border-b border-border">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setIsSidebarOpen(true)}
              className="text-sky-500 hover:text-sky-400 hover:bg-sky-500/10"
            >
              <Menu className="w-5 h-5" />
            </Button>
            <span className="text-sm font-medium text-muted-foreground select-none">
              {sectionLabel}
            </span>
          </div>
        </div>
      </header>

      {/* Spacer to offset the fixed header height */}
      <div className="h-16 flex-shrink-0" />

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed top-0 left-0 h-full w-56 bg-card border-r border-border z-[2500] transition-transform duration-300",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
        ref={menuRef}
        aria-label={t('common.menu', { defaultValue: 'Menu' })}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-border">
          <span className="text-xs text-muted-foreground">{t('common.navigation', { defaultValue: 'Navigation' })}</span>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={t('common.close', { defaultValue: 'Close' })}
            onClick={() => setIsSidebarOpen(false)}
            className="text-muted-foreground hover:text-foreground hover:bg-muted/50"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        <nav className="flex flex-col p-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;

            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveSection(item.id);
                  setIsSidebarOpen(false);
                }}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors",
                  isActive
                    ? "bg-muted text-primary font-semibold"
                    : "text-muted-foreground hover:bg-muted/70"
                )}
              >
                <Icon className="w-5 h-5" />
                {t(item.labelKey)}
              </button>
            );
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <main
        className={cn(
          "flex-1 overflow-hidden relative transition-all duration-300 ease-in-out",
          isSidebarOpen ? "sm:ml-64" : "ml-0"
        )}
      >
        {/* MAP PANEL */}
        <div
          role="tabpanel"
          id="map-panel"
          aria-labelledby="map-tab"
          className={cn(
            "absolute inset-0 h-full w-full transition-opacity duration-300",
            activeSection === 'map'
              ? "z-10 opacity-100"
              : "z-0 opacity-0 pointer-events-none"
          )}
        >
          <CatchMap className="h-full w-full" />
        </div>

        {/* NEWS PANEL */}
        <div
          role="tabpanel"
          id="news-panel"
          aria-labelledby="news-tab"
          className={cn(
            "absolute inset-0 h-full w-full overflow-hidden transition-opacity duration-300",
            activeSection === 'news'
              ? "z-10 opacity-100"
              : "z-0 opacity-0 pointer-events-none"
          )}
        >
          <NewsModule />
        </div>

        {/* WEATHER PANEL */}
        <div
          role="tabpanel"
          id="weather-panel"
          aria-labelledby="weather-tab"
          className={cn(
            "absolute inset-0 h-full w-full overflow-hidden transition-opacity duration-300",
            activeSection === 'weather'
              ? "z-10 opacity-100"
              : "z-0 opacity-0 pointer-events-none"
          )}
        >
          <WeatherModule />
        </div>
      </main>
    </div>
  );
}
