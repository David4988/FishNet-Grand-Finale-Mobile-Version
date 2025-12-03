import { useState, useEffect } from "react";
import { Wifi, WifiOff, Cloud, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { syncService, SyncStatus } from "@/services/sync";
import { cn } from "@/lib/utils";

export const OfflineIndicator = () => {
  const [syncStatus, setSyncStatus] = useState<SyncStatus>(syncService.getStatus());
  // This state now specifically controls the temporary "back online" message
  const [showOnlineBanner, setShowOnlineBanner] = useState(false);

  useEffect(() => {
    const unsubscribe = syncService.subscribe((newStatus) => {
      setSyncStatus(prevStatus => {
        // --- KEY LOGIC ---
        // If we just came online from being offline...
        if (newStatus.isOnline && !prevStatus.isOnline) {
          setShowOnlineBanner(true); // Show the banner
          // Set a timer to hide it after 2 seconds
          setTimeout(() => setShowOnlineBanner(false), 2000); 
        }
        return newStatus;
      });
    });

    return unsubscribe;
  }, []); // Empty array is correct, our logic handles stale state

  const handleSync = async () => {
    if (syncStatus.isOnline && !syncStatus.isSyncing) {
      await syncService.forcSync();
    }
  };

  const hasPendingItems = syncStatus.pendingCatches > 0 || syncStatus.pendingPosts > 0;

  // --- NEW RENDER LOGIC ---
  // Render ONLY if we are offline OR if the temporary online banner is active.
  if (syncStatus.isOnline && !showOnlineBanner) {
    return null;
  }

  return (
    <div className={cn(
      "fixed top-0 left-0 right-0 z-50 transition-transform duration-300",
      !syncStatus.isOnline || showOnlineBanner ? "translate-y-0" : "-translate-y-full"
    )}>
      <Card className={cn(
        "mx-4 mt-4 border-0 shadow-lg",
        syncStatus.isOnline 
          ? "bg-success/10 border-success/20"  // "Back Online" state
          : "bg-warning/10 border-warning/20"  // "Offline" state
      )}>
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {syncStatus.isOnline ? (
                <Wifi className="w-4 h-4 text-success" />
              ) : (
                <WifiOff className="w-4 h-4 text-warning" />
              )}
              <span className={cn(
                "text-sm font-medium",
                syncStatus.isOnline ? "text-success" : "text-warning"
              )}>
                {syncStatus.isOnline ? "Back Online" : "Offline Mode"}
              </span>
              
              {syncStatus.isSyncing && (
                <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
              )}
            </div>
            
            {hasPendingItems && !syncStatus.isOnline && (
               <Badge 
                variant="secondary"
                className="bg-blue-100 text-blue-700 border-blue-200 text-xs"
              >
                {syncStatus.pendingCatches + syncStatus.pendingPosts} pending
              </Badge>
            )}
          </div>

          {/* Details are only shown when offline and there are pending items */}
          {hasPendingItems && !syncStatus.isOnline && (
            <div className="mt-3 pt-3 border-t border-border/50">
              <div className="flex items-center justify-between">
                <div className="text-xs text-muted-foreground space-y-1">
                  {syncStatus.pendingCatches > 0 && <div>📸 {syncStatus.pendingCatches} catches waiting</div>}
                  {syncStatus.pendingPosts > 0 && <div>📝 {syncStatus.pendingPosts} posts waiting</div>}
                </div>
                {/* No sync button when offline */}
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};