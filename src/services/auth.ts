// Fake Auth Service for Offline PWA
// Simulates authentication with local storage for offline-first experience

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  bio?: string;
  bioKey?: string;
  joinDate: string;
  totalCatches: number;
  favoriteSpecies: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  loading: boolean;
}

class AuthService {
  private storageKey = 'fishnet_auth';
  private listeners: ((state: AuthState) => void)[] = [];
  private currentState: AuthState = {
    isAuthenticated: false,
    user: null,
    loading: true
  };

  constructor() {
    this.loadFromStorage();
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const { user } = JSON.parse(stored);
        this.currentState = {
          isAuthenticated: true,
          user,
          loading: false
        };
      } else {
        this.currentState.loading = false;
      }
    } catch (error) {
      console.warn('Failed to load auth from storage:', error);
      this.currentState.loading = false;
    }
    this.notifyListeners();
  }

  private saveToStorage(user: User) {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify({ user }));
    } catch (error) {
      console.warn('Failed to save auth to storage:', error);
    }
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.currentState));
  }

  getState(): AuthState {
    return { ...this.currentState };
  }

  subscribe(listener: (state: AuthState) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  async signUp(email: string, password: string, name: string): Promise<{ success: boolean; error?: string }> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Check if user already exists
    const existingUsers = this.getStoredUsers();
    if (existingUsers.some(u => u.email === email)) {
      return { success: false, error: 'User with this email already exists' };
    }

    // Create new user
    const user: User = {
      id: `user_${Date.now()}`,
      email,
      name,
      bioKey: 'profile.defaultBio',
      joinDate: new Date().toISOString(),
      totalCatches: 0,
      favoriteSpecies: 'Unknown',
      avatar: `https://api.dicebear.com/7.x/adventurer/svg?seed=${email}`
    };

    // Save user to fake "database"
    this.saveUser(user);

    // Set as current user
    this.currentState = {
      isAuthenticated: true,
      user,
      loading: false
    };

    this.saveToStorage(user);
    this.notifyListeners();

    return { success: true };
  }

  async signIn(email: string, password: string): Promise<{ success: boolean; error?: string }> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Find user in fake "database"
    const existingUsers = this.getStoredUsers();
    const user = existingUsers.find(u => u.email === email);

    if (!user) {
      return { success: false, error: 'No account found with this email' };
    }

    // Set as current user
    this.currentState = {
      isAuthenticated: true,
      user,
      loading: false
    };

    this.saveToStorage(user);
    this.notifyListeners();

    return { success: true };
  }

  async signOut(): Promise<void> {
    localStorage.removeItem(this.storageKey);
    this.currentState = {
      isAuthenticated: false,
      user: null,
      loading: false
    };
    this.notifyListeners();
  }

  async updateProfile(updates: Partial<Pick<User, 'name' | 'bio' | 'avatar'>>): Promise<{ success: boolean; error?: string }> {
    if (!this.currentState.user) {
      return { success: false, error: 'Not authenticated' };
    }

    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));

    const updatedUser = { ...this.currentState.user, ...updates };
    
    // Update in fake "database"
    this.saveUser(updatedUser);

    // Update current state
    this.currentState.user = updatedUser;
    this.saveToStorage(updatedUser);
    this.notifyListeners();

    return { success: true };
  }

  private getStoredUsers(): User[] {
    try {
      const stored = localStorage.getItem('fishnet_users');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  private saveUser(user: User) {
    try {
      const users = this.getStoredUsers();
      const index = users.findIndex(u => u.id === user.id);
      if (index >= 0) {
        users[index] = user;
      } else {
        users.push(user);
      }
      localStorage.setItem('fishnet_users', JSON.stringify(users));
    } catch (error) {
      console.warn('Failed to save user:', error);
    }
  }

  // Get demo users for social feed
  getDemoUsers(): User[] {
    const demoUsers: User[] = [
      {
        id: 'demo_1',
        email: 'captain.hook@fishnet.com',
        name: 'Captain Hook',
        bio: 'Professional angler with 20+ years experience. Specialist in deep sea fishing.',
        joinDate: '2024-01-15T00:00:00Z',
        totalCatches: 847,
        favoriteSpecies: 'Marlin (Blue)',
        avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=captain'
      },
      {
        id: 'demo_2',
        email: 'marina.fisher@fishnet.com',
        name: 'Marina Fisher',
        bio: '🎣 Weekend warrior | 🌊 Ocean lover | 📸 Catch photographer',
        joinDate: '2024-02-20T00:00:00Z',
        totalCatches: 234,
        favoriteSpecies: 'Salmon (Atlantic)',
        avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=marina'
      },
      {
        id: 'demo_3',
        email: 'reef.explorer@fishnet.com',
        name: 'Reef Explorer',
        bio: 'Marine biologist and conservation enthusiast. Promoting sustainable fishing.',
        joinDate: '2024-03-10T00:00:00Z',
        totalCatches: 156,
        favoriteSpecies: 'Grouper (Nassau)',
        avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=reef'
      }
    ];

    // Ensure demo users are saved
    demoUsers.forEach(user => this.saveUser(user));
    return demoUsers;
  }
}

export const authService = new AuthService();