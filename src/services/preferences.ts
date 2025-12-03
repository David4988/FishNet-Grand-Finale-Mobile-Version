// User Preferences Service
// Manages user settings and device permissions

export interface UserPreferences {
  language: string;
  locationEnabled: boolean;
  locationPermissionState: 'granted' | 'denied' | 'prompt' | 'unknown';
}

class PreferencesService {
  private storageKey = 'fishnet_preferences';
  private listeners: ((prefs: UserPreferences) => void)[] = [];
  private currentPrefs: UserPreferences = {
    language: 'en',
    locationEnabled: false,
    locationPermissionState: 'unknown'
  };

  constructor() {
    this.loadPreferences();
    this.checkLocationPermission();
  }

  private loadPreferences() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        this.currentPrefs = { ...this.currentPrefs, ...JSON.parse(stored) };
      }
    } catch (error) {
      console.warn('Failed to load preferences:', error);
    }
  }

  private savePreferences() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.currentPrefs));
      this.notifyListeners();
    } catch (error) {
      console.warn('Failed to save preferences:', error);
    }
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener({ ...this.currentPrefs }));
  }

  private async checkLocationPermission() {
    if (!navigator.permissions || !navigator.permissions.query) {
      this.currentPrefs.locationPermissionState = 'unknown';
      return;
    }

    try {
      const result = await navigator.permissions.query({ name: 'geolocation' as PermissionName });
      this.currentPrefs.locationPermissionState = result.state as 'granted' | 'denied' | 'prompt';
      
      result.addEventListener('change', () => {
        this.currentPrefs.locationPermissionState = result.state as 'granted' | 'denied' | 'prompt';
        this.notifyListeners();
      });
    } catch (error) {
      console.warn('Failed to query location permission:', error);
      this.currentPrefs.locationPermissionState = 'unknown';
    }
  }

  getPreferences(): UserPreferences {
    return { ...this.currentPrefs };
  }

  subscribe(listener: (prefs: UserPreferences) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  async setLanguage(language: string): Promise<void> {
    this.currentPrefs.language = language;
    localStorage.setItem('fishnet_language', language);
    this.savePreferences();
  }

  async requestLocationPermission(): Promise<{ success: boolean; position?: GeolocationPosition; error?: string }> {
    if (!navigator.geolocation) {
      return { success: false, error: 'Geolocation is not supported by this browser' };
    }

    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          this.currentPrefs.locationEnabled = true;
          this.currentPrefs.locationPermissionState = 'granted';
          this.savePreferences();
          resolve({ success: true, position });
        },
        (error) => {
          this.currentPrefs.locationEnabled = false;
          this.currentPrefs.locationPermissionState = 'denied';
          this.savePreferences();
          resolve({ success: false, error: error.message });
        },
        {
          enableHighAccuracy: true,
          timeout: 5000,
          maximumAge: 0
        }
      );
    });
  }

  async disableLocation(): Promise<void> {
    this.currentPrefs.locationEnabled = false;
    this.savePreferences();
  }

  async getCurrentLocation(): Promise<{ success: boolean; position?: GeolocationPosition; error?: string }> {
    if (!this.currentPrefs.locationEnabled || !navigator.geolocation) {
      return { success: false, error: 'Location access is disabled' };
    }

    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({ success: true, position });
        },
        (error) => {
          resolve({ success: false, error: error.message });
        }
      );
    });
  }
}

export const preferencesService = new PreferencesService();
