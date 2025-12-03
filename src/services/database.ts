// Offline SQLite Database Service for Fish Catches
import { openDB, IDBPDatabase } from 'idb';

// Constants for database and store names
const DB_NAME = 'fish-scan-db';
const CATCHES_STORE = 'catches';
const IMAGES_STORE = 'images';

// Simplified in-memory storage as fallback
class SimpleDatabaseService {
  private catches: (FishCatch & { id: number })[] = [];
  private nextId = 1;

  async initialize() {
    console.log('Using fallback in-memory database');
  }

  // Image storage methods
  async storeImage(imageId: string, blob: Blob): Promise<void> {
    // In-memory storage doesn't support image storage
    console.warn('Image storage not supported in fallback mode');
  }

  async getImage(imageId: string): Promise<Blob | null> {
    // In-memory storage doesn't support image storage
    console.warn('Image storage not supported in fallback mode');
    return null;
  }

  async addCatch(catch_data: Omit<FishCatch, 'id'>): Promise<number> {
    const id = this.nextId++;
    this.catches.push({ ...catch_data, id });
    return id;
  }

  async getAllCatches(): Promise<FishCatch[]> {
    return [...this.catches].sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }

  async getCatchesBySpecies(species: string): Promise<FishCatch[]> {
    return this.catches.filter(c => c.species === species)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  async getUnsyncedCatches(): Promise<FishCatch[]> {
    return this.catches.filter(c => !c.is_synced);
  }

  async markAsSynced(id: number) {
    const catch_data = this.catches.find(c => c.id === id);
    if (catch_data) catch_data.is_synced = true;
  }

  async deleteCatch(id: number) {
    const index = this.catches.findIndex(c => c.id === id);
    if (index >= 0) this.catches.splice(index, 1);
  }

  async getStats() {
    return {
      totalCatches: this.catches.length,
      uniqueSpecies: new Set(this.catches.map(c => c.species)).size,
      unsyncedCount: this.catches.filter(c => !c.is_synced).length
    };
  }
}

export interface FishCatch {
  id?: number;
  species: string;
  confidence: number;
  health_score: number;
  count: number;
  estimated_weight: number;
  timestamp: string;
  latitude: number;
  longitude: number;
  image_data: string;
  is_synced: boolean;
}

class DatabaseService {
  private fallbackService = new SimpleDatabaseService();
  private db: IDBPDatabase | null = null;

  async initialize() {
    try {
      this.db = await openDB(DB_NAME, 1, {
        upgrade(db) {
          // Create catches store if it doesn't exist
          if (!db.objectStoreNames.contains(CATCHES_STORE)) {
            db.createObjectStore(CATCHES_STORE, { keyPath: 'id', autoIncrement: true });
          }
          // Create images store if it doesn't exist
          if (!db.objectStoreNames.contains(IMAGES_STORE)) {
            db.createObjectStore(IMAGES_STORE, { keyPath: 'id' });
          }
        },
      });
      console.log('IndexedDB database service initialized');
    } catch (error) {
      console.error('IndexedDB initialization failed, falling back to memory storage:', error);
      await this.fallbackService.initialize();
    }
  }

  async storeImage(imageId: string, blob: Blob): Promise<void> {
    if (!this.db) {
      return this.fallbackService.storeImage(imageId, blob);
    }
    try {
      const tx = this.db.transaction(IMAGES_STORE, 'readwrite');
      await tx.store.put({ id: imageId, blob });
      await tx.done;
    } catch (error) {
      console.error('Failed to store image:', error);
      throw error;
    }
  }

  async getImage(imageId: string): Promise<Blob | null> {
    if (!this.db) {
      return this.fallbackService.getImage(imageId);
    }
    try {
      const tx = this.db.transaction(IMAGES_STORE, 'readonly');
      const result = await tx.store.get(imageId);
      return result?.blob || null;
    } catch (error) {
      console.error('Failed to retrieve image:', error);
      return null;
    }
  }

  async addCatch(catch_data: Omit<FishCatch, 'id'>): Promise<number> {
    if (!this.db) {
      return this.fallbackService.addCatch(catch_data);
    }
    
    try {
      // If there's image data as base64, convert and store it
      if (catch_data.image_data && catch_data.image_data.startsWith('data:image')) {
        const response = await fetch(catch_data.image_data);
        const blob = await response.blob();
        const imageId = `img_${Date.now()}`;
        await this.storeImage(imageId, blob);
        catch_data.image_data = imageId; // Store only the image ID
      }

      const tx = this.db.transaction(CATCHES_STORE, 'readwrite');
      const id = await tx.store.add(catch_data);
      await tx.done;
      return typeof id === 'number' ? id : parseInt(id.toString(), 10);
    } catch (error) {
      console.error('Failed to add catch:', error);
      return this.fallbackService.addCatch(catch_data);
    }
  }

  async getAllCatches(): Promise<FishCatch[]> {
    if (!this.db) {
      return this.fallbackService.getAllCatches();
    }
    try {
      const tx = this.db.transaction(CATCHES_STORE, 'readonly');
      const all = await tx.store.getAll();
      // Sort newest first
      return all.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    } catch (error) {
      console.error('Failed to get all catches:', error);
      return this.fallbackService.getAllCatches();
    }
  }

  async getCatchesBySpecies(species: string): Promise<FishCatch[]> {
    return this.fallbackService.getCatchesBySpecies(species);
  }

  async getUnsyncedCatches(): Promise<FishCatch[]> {
    return this.fallbackService.getUnsyncedCatches();
  }

  async markAsSynced(id: number) {
    return this.fallbackService.markAsSynced(id);
  }

  async deleteCatch(id: number) {
    return this.fallbackService.deleteCatch(id);
  }

  async getStats() {
    return this.fallbackService.getStats();
  }
}

export const databaseService = new DatabaseService();