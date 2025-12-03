// Enhanced Social Feed Service with Real-time Updates
import { User } from './auth';

export type { User };

export interface SocialPost {
  id: string;
  userId: string;
  user: User;
  species: string;
  confidence: number;
  healthScore: number;
  estimatedWeight: number;
  estimatedCount: number;
  imageData: string;
  caption: string;
  location?: {
    latitude: number;
    longitude: number;
    name?: string;
  };
  timestamp: string;
  likes: number;
  likedBy: string[];
  comments: Comment[];
  isOffline?: boolean;
}

export interface Comment {
  id: string;
  userId: string;
  user: User;
  text: string;
  timestamp: string;
}

class SocialService {
  private storageKey = 'fishnet_posts';
  private likesKey = 'fishnet_likes';
  private listeners: (() => void)[] = [];

  constructor() {
    this.initializeDemoData();
  }

  private initializeDemoData() {
    // Only add demo data if none exists
    const posts = this.getAllPosts();
    if (posts.length === 0) {
      this.createDemoPosts();
    }
  }

  private createDemoPosts() {
    const demoPosts: SocialPost[] = [
      {
        id: 'post_1',
        userId: 'demo_1',
        user: {
          id: 'demo_1',
          email: 'captain.hook@fishnet.com',
          name: 'Captain Hook',
          bio: 'Professional angler with 20+ years experience.',
          joinDate: '2024-01-15T00:00:00Z',
          totalCatches: 47,
          favoriteSpecies: 'Marlin (Blue)',
          avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=captain'
        },
        species: 'Marlin (Blue)',
        confidence: 94,
        healthScore: 88,
        estimatedWeight: 5.2,
        estimatedCount: 1,
        imageData: 'https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=400&h=300&fit=crop',
        caption: 'What a fight! This blue marlin gave me the battle of a lifetime. 5kg of pure power! 🎣⚡ #BigGameFishing #MarlinMadness',
        location: {
          latitude: 25.7617,
          longitude: -80.1918,
          name: 'Miami Beach, FL'
        },
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        likes: 127,
        likedBy: ['demo_2', 'demo_3'],
        comments: [
          {
            id: 'comment_1',
            userId: 'demo_2',
            user: {
              id: 'demo_2',
              name: 'Marina Fisher',
              avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=marina'
            } as User,
            text: 'Incredible catch! The fight must have been amazing 🔥',
            timestamp: new Date(Date.now() - 1.5 * 60 * 60 * 1000).toISOString()
          }
        ]
      },
      {
        id: 'post_2',
        userId: 'demo_2',
        user: {
          id: 'demo_2',
          email: 'marina.fisher@fishnet.com',
          name: 'Marina Fisher',
          bio: '🎣 Weekend warrior | 🌊 Ocean lover',
          joinDate: '2024-02-20T00:00:00Z',
          totalCatches: 234,
          favoriteSpecies: 'Salmon (Atlantic)',
          avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=marina'
        },
        species: 'Salmon (Atlantic)',
        confidence: 89,
        healthScore: 92,
        estimatedWeight: 8.1,
        estimatedCount: 1,
        imageData: 'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400&h=300&fit=crop',
        caption: 'Perfect morning on the river! This Atlantic salmon was so beautiful, had to snap a pic before release 📸✨ #CatchAndRelease #SalmonFishing',
        location: {
          latitude: 44.2619,
          longitude: -72.5806,
          name: 'Vermont River'
        },
        timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
        likes: 89,
        likedBy: ['demo_1'],
        comments: [
          {
            id: 'comment_3',
            userId: 'demo_1',
            user: {
              id: 'demo_1',
              name: 'Captain Hook',
              avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=captain'
            } as User,
            text: 'Beautiful release! Respect for conservation 🌊',
            timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString()
          }
        ]
      },
      {
        id: 'post_3',
        userId: 'demo_3',
        user: {
          id: 'demo_3',
          email: 'reef.explorer@fishnet.com',
          name: 'Reef Explorer',
          bio: 'Marine biologist and conservation enthusiast.',
          joinDate: '2024-03-10T00:00:00Z',
          totalCatches: 156,
          favoriteSpecies: 'Grouper (Nassau)',
          avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=reef'
        },
        species: 'Grouper (Nassau)',
        confidence: 91,
        healthScore: 85,
        estimatedWeight: 12.7,
        estimatedCount: 1,
        imageData: 'https://images.unsplash.com/photo-1583212292454-1fe6229603b7?w=400&h=300&fit=crop',
        caption: 'Nassau grouper spotted in the coral reef! These beautiful fish are critical to reef ecosystems 🐠🪸 #MarineBiology #ReefConservation',
        location: {
          latitude: 25.0343,
          longitude: -77.3963,
          name: 'Bahamas Reef'
        },
        timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
        likes: 156,
        likedBy: ['demo_1', 'demo_2'],
        comments: [
          {
            id: 'comment_2',
            userId: 'demo_1',
            user: {
              id: 'demo_1',
              name: 'Captain Hook',
              avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=captain'
            } as User,
            text: 'Great shot! Love seeing healthy reef systems 🌊',
            timestamp: new Date(Date.now() - 7 * 60 * 60 * 1000).toISOString()
          },
          {
            id: 'comment_4',
            userId: 'demo_2',
            user: {
              id: 'demo_2',
              name: 'Marina Fisher',
              avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=marina'
            } as User,
            text: 'The AI analysis on this is impressive! 91% confidence 🤖',
            timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString()
          }
        ]
      },
      {
        id: 'post_4',
        userId: 'demo_1',
        user: {
          id: 'demo_1',
          email: 'captain.hook@fishnet.com',
          name: 'Captain Hook',
          bio: 'Professional angler with 20+ years experience.',
          joinDate: '2024-01-15T00:00:00Z',
          totalCatches: 847,
          favoriteSpecies: 'Marlin (Blue)',
          avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=captain'
        },
        species: 'Tuna (Yellowfin)',
        confidence: 87,
        healthScore: 90,
        estimatedWeight: 28.4,
        estimatedCount: 1,
        imageData: 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=300&fit=crop',
        caption: 'Yellowfin tuna from yesterday\'s charter! Sashimi grade quality 🍣 The AI health score of 90% confirms this is premium grade fish. #TunaFishing #FreshCatch #SashimiGrade',
        location: {
          latitude: 32.7157,
          longitude: -117.1611,
          name: 'San Diego, CA'
        },
        timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        likes: 203,
        likedBy: ['demo_2', 'demo_3'],
        comments: [
          {
            id: 'comment_5',
            userId: 'demo_3',
            user: {
              id: 'demo_3',
              name: 'Reef Explorer',
              avatar: 'https://api.dicebear.com/7.x/adventurer/svg?seed=reef'
            } as User,
            text: 'That health score is incredible! Perfect for sashimi 🍣',
            timestamp: new Date(Date.now() - 20 * 60 * 60 * 1000).toISOString()
          }
        ]
      }
    ];

    this.savePosts(demoPosts);
  }

  getAllPosts(): SocialPost[] {
    try {
      const stored = localStorage.getItem(this.storageKey);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  private savePosts(posts: SocialPost[]) {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(posts));
      this.notifyListeners();
    } catch (error) {
      console.warn('Failed to save posts:', error);
    }
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener());
  }

  subscribe(listener: () => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  async createPost(post: Omit<SocialPost, 'id' | 'likes' | 'likedBy' | 'comments'>): Promise<SocialPost> {
    const newPost: SocialPost = {
      ...post,
      id: `post_${Date.now()}`,
      likes: 0,
      likedBy: [],
      comments: [],
      isOffline: !navigator.onLine
    };

    const posts = this.getAllPosts();
    posts.unshift(newPost); // Add to beginning for newest first
    this.savePosts(posts);

    return newPost;
  }

  async toggleLike(postId: string, userId: string): Promise<boolean> {
    const posts = this.getAllPosts();
    const postIndex = posts.findIndex(p => p.id === postId);
    
    if (postIndex === -1) return false;

    const post = posts[postIndex];
    const isLiked = post.likedBy.includes(userId);

    if (isLiked) {
      post.likedBy = post.likedBy.filter(id => id !== userId);
      post.likes = Math.max(0, post.likes - 1);
    } else {
      post.likedBy.push(userId);
      post.likes += 1;
    }

    this.savePosts(posts);
    return !isLiked;
  }

  async addComment(postId: string, userId: string, user: User, text: string): Promise<Comment> {
    const comment: Comment = {
      id: `comment_${Date.now()}`,
      userId,
      user,
      text,
      timestamp: new Date().toISOString()
    };

    const posts = this.getAllPosts();
    const postIndex = posts.findIndex(p => p.id === postId);
    
    if (postIndex !== -1) {
      posts[postIndex].comments.push(comment);
      this.savePosts(posts);
    }

    return comment;
  }

  getPostsByUser(userId: string): SocialPost[] {
    return this.getAllPosts().filter(post => post.userId === userId);
  }

  getFeedPosts(limit = 20): SocialPost[] {
    return this.getAllPosts()
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  isPostLiked(postId: string, userId: string): boolean {
    const posts = this.getAllPosts();
    const post = posts.find(p => p.id === postId);
    return post ? post.likedBy.includes(userId) : false;
  }

  getPostsForMap(): Array<SocialPost & { location: NonNullable<SocialPost['location']> }> {
    return this.getAllPosts()
      .filter((post): post is SocialPost & { location: NonNullable<SocialPost['location']> } => 
        post.location !== undefined
      );
  }

  // Real-time updates
  updatePostInRealTime(postId: string, updates: Partial<SocialPost>) {
    const posts = this.getAllPosts();
    const postIndex = posts.findIndex(p => p.id === postId);
    
    if (postIndex !== -1) {
      posts[postIndex] = { ...posts[postIndex], ...updates };
      this.savePosts(posts);
    }
  }

  // Get posts that need syncing
  getUnsyncedPosts(): SocialPost[] {
    return this.getAllPosts().filter(post => post.isOffline);
  }

  // Mark posts as synced
  markPostAsSynced(postId: string) {
    const posts = this.getAllPosts();
    const postIndex = posts.findIndex(p => p.id === postId);
    
    if (postIndex !== -1) {
      posts[postIndex].isOffline = false;
      this.savePosts(posts);
    }
  }
}

export const socialService = new SocialService();