import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { MobileLayout } from "@/components/layout/MobileLayout";
import { BottomNavigation } from "@/components/layout/BottomNavigation";
import { OfflineIndicator } from "@/components/offline/OfflineIndicator";
import FeedPage from "./pages/FeedPage";
import AnalyzePage from "./pages/AnalyzePage";
import MapPage from "./pages/MapPage";
import HistoryPage from "./pages/HistoryPage";
import ProfilePage from "./pages/ProfilePage";
import SpeciesPage from "./pages/SpeciesPage";
import NotFound from "./pages/NotFound";
import { seedSampleData } from "@/services/sampleData";
import { pwaService } from "@/services/pwa";
import { syncService } from "@/services/sync";
import { useEffect } from "react";

const queryClient = new QueryClient();

const App = () => {
  useEffect(() => {
    // Initialize app services
    const initializeApp = async () => {
      try {
        // Seed sample data
        await seedSampleData();
        
        // Register PWA service worker
        if ('serviceWorker' in navigator) {
          await pwaService.register();
        }
        
        // Initialize sync service (already done in constructor)
        console.log('Fish Net app initialized successfully');
      } catch (error) {
        console.warn('App initialization warning:', error);
      }
    };

    initializeApp();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Toaster />
      <Sonner />
      <OfflineIndicator />
      <AuthGuard>
        <BrowserRouter>
          <MobileLayout>
            <Routes>
              <Route path="/" element={<FeedPage />} />
              <Route path="/analyze" element={<AnalyzePage />} />
              <Route path="/map" element={<MapPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="/species" element={<SpeciesPage />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
            <BottomNavigation />
          </MobileLayout>
        </BrowserRouter>
      </AuthGuard>
    </QueryClientProvider>
  );
};

export default App;