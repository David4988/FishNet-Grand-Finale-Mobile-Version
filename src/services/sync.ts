// Offline/Online Sync Service for FishNet
import { databaseService, FishCatch } from './database';
import { socialService, SocialPost } from './social';

export interface SyncStatus {
  isOnline: boolean;
  pendingCatches: number;
  pendingPosts: number;
  lastSyncTime: string | null;
  isSyncing: boolean;
}

class SyncService {
  private listeners: ((status: SyncStatus) => void)[] = [];
  private syncStatus: SyncStatus = {
    isOnline: navigator.onLine,
    pendingCatches: 0,
    pendingPosts: 0,
    lastSyncTime: null,
    isSyncing: false
  };

  constructor() {
    this.initializeSync();
    this.loadSyncStatus();
  }

  private initializeSync() {
    // Listen for online/offline events
    window.addEventListener('online', this.handleOnline.bind(this));
    window.addEventListener('offline', this.handleOffline.bind(this));

    // Check sync status periodically
    setInterval(() => {
      this.updatePendingCounts();
    }, 5000);

    // Initial status update
    this.updatePendingCounts();
  }

  private loadSyncStatus() {
    try {
      const stored = localStorage.getItem('fishnet_sync_status');
      if (stored) {
        const data = JSON.parse(stored);
        this.syncStatus.lastSyncTime = data.lastSyncTime;
      }
    } catch (error) {
      console.warn('Failed to load sync status:', error);
    }
  }

  private saveSyncStatus() {
    try {
      localStorage.setItem('fishnet_sync_status', JSON.stringify({
        lastSyncTime: this.syncStatus.lastSyncTime
      }));
    } catch (error) {
      console.warn('Failed to save sync status:', error);
    }
  }

  private async updatePendingCounts() {
    try {
      const unsyncedCatches = await databaseService.getUnsyncedCatches();
      const unsyncedPosts = this.getUnsyncedPosts();
      
      this.syncStatus.pendingCatches = unsyncedCatches.length;
      this.syncStatus.pendingPosts = unsyncedPosts.length;
      
      this.notifyListeners();
    } catch (error) {
      console.warn('Failed to update pending counts:', error);
    }
  }

  private getUnsyncedPosts(): SocialPost[] {
    // Get posts that were created offline
    const allPosts = socialService.getAllPosts();
    return allPosts.filter(post => post.isOffline);
  }

  private async handleOnline() {
    console.log('Device came online - starting sync...');
    this.syncStatus.isOnline = true;
    this.notifyListeners();
    
    // Start automatic sync after a short delay
    setTimeout(() => {
      this.syncAll();
    }, 2000);
  }

  private handleOffline() {
    console.log('Device went offline');
    this.syncStatus.isOnline = false;
    this.syncStatus.isSyncing = false;
    this.notifyListeners();
  }

  async syncAll(): Promise<{ success: boolean; error?: string }> {
    if (!this.syncStatus.isOnline) {
      return { success: false, error: 'Device is offline' };
    }

    if (this.syncStatus.isSyncing) {
      return { success: false, error: 'Sync already in progress' };
    }

    this.syncStatus.isSyncing = true;
    this.notifyListeners();

    try {
      console.log('Starting full sync...');
      
      // Sync catches
      const catchResult = await this.syncCatches();
      if (!catchResult.success) {
        throw new Error(catchResult.error);
      }

      // Sync posts
      const postResult = await this.syncPosts();
      if (!postResult.success) {
        throw new Error(postResult.error);
      }

      // Update sync time
      this.syncStatus.lastSyncTime = new Date().toISOString();
      this.saveSyncStatus();

      console.log('Sync completed successfully');
      return { success: true };

    } catch (error) {
      console.error('Sync failed:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Sync failed' };
    } finally {
      this.syncStatus.isSyncing = false;
      await this.updatePendingCounts();
      this.notifyListeners();
    }
  }

  private async syncCatches(): Promise<{ success: boolean; error?: string }> {
    try {
      const unsyncedCatches = await databaseService.getUnsyncedCatches();
      console.log(`Syncing ${unsyncedCatches.length} catches...`);

      for (const catch_data of unsyncedCatches) {
        // Simulate API call to sync catch
        await this.simulateApiCall(`/api/catches`, catch_data);
        
        // Mark as synced
        if (catch_data.id) {
          await databaseService.markAsSynced(catch_data.id);
        }
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Failed to sync catches' };
    }
  }

  private async syncPosts(): Promise<{ success: boolean; error?: string }> {
    try {
      const unsyncedPosts = this.getUnsyncedPosts();
      console.log(`Syncing ${unsyncedPosts.length} posts...`);

      for (const post of unsyncedPosts) {
        // Simulate API call to sync post
        await this.simulateApiCall(`/api/posts`, post);
        
        // Mark as synced (remove offline flag)
        post.isOffline = false;
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Failed to sync posts' };
    }
  }

  private async simulateApiCall(endpoint: string, data: any): Promise<void> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
    
    // Simulate occasional failures for realism
    if (Math.random() < 0.1) {
      throw new Error('Network error during sync');
    }
    
    console.log(`Synced to ${endpoint}:`, data.species || data.id);
  }

  getStatus(): SyncStatus {
    return { ...this.syncStatus };
  }

  subscribe(listener: (status: SyncStatus) => void): () => void {
    this.listeners.push(listener);
    // Immediately notify with current status
    listener(this.syncStatus);
    
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener({ ...this.syncStatus }));
  }

  async forcSync(): Promise<{ success: boolean; error?: string }> {
    return this.syncAll();
  }

  async clearSyncData() {
    try {
      localStorage.removeItem('fishnet_sync_status');
      this.syncStatus.lastSyncTime = null;
      this.syncStatus.pendingCatches = 0;
      this.syncStatus.pendingPosts = 0;
      this.notifyListeners();
    } catch (error) {
      console.warn('Failed to clear sync data:', error);
    }
  }
}

export const syncService = new SyncService();