// Chat service for managing conversations and messages
import { authService, User } from './auth';

export interface ChatMessage {
  id: string;
  conversationId: string;
  senderId: string;
  senderName: string;
  senderAvatar: string;
  text: string;
  timestamp: string;
  isRead: boolean;
  isOffline?: boolean;
}

export interface Conversation {
  id: string;
  otherUser: User;
  lastMessage?: ChatMessage;
  unreadCount: number;
  updatedAt: string;
}

class ChatService {
  private conversations: Map<string, Conversation> = new Map();
  private messages: Map<string, ChatMessage[]> = new Map();
  private listeners: Set<() => void> = new Set();

  constructor() {
    this.initializeSampleData();
  }

  initialize() {
    this.conversations.clear();
    this.messages.clear();
    this.initializeSampleData();
    this.notifyListeners();
  }

  private initializeSampleData() {
    const currentUser = authService.getState().user;
    if (!currentUser) return;

    // Sample fishermen to chat with
    const sampleUsers: User[] = [
      {
        id: 'user2',
        email: 'raj@fishnet.com',
        name: 'Raj Kumar',
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=raj',
        bio: 'Professional fisherman from Mumbai',
        joinDate: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString(),
        totalCatches: 156,
        favoriteSpecies: 'Pomfret'
      },
      {
        id: 'user3',
        email: 'priya@fishnet.com',
        name: 'Priya Sharma',
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=priya',
        bio: 'Marine biologist and fishing enthusiast',
        joinDate: new Date(Date.now() - 120 * 24 * 60 * 60 * 1000).toISOString(),
        totalCatches: 89,
        favoriteSpecies: 'Kingfish'
      },
      {
        id: 'user4',
        email: 'arjun@fishnet.com',
        name: 'Arjun Patel',
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=arjun',
        bio: 'Deep sea fishing expert from Goa',
        joinDate: new Date(Date.now() - 240 * 24 * 60 * 60 * 1000).toISOString(),
        totalCatches: 203,
        favoriteSpecies: 'Tuna'
      }
    ];

    // Store messages in localStorage for caching
    const cachedMessages = localStorage.getItem('fishnet_chat_messages');
    if (cachedMessages) {
      try {
        const parsed = JSON.parse(cachedMessages);
        this.messages = new Map(Object.entries(parsed));
      } catch (e) {
        console.warn('Failed to load cached messages');
      }
    }

    // Create conversations
    sampleUsers.forEach((user, index) => {
      const conversationId = `conv_${user.id}`;
      const lastMessageTime = new Date(Date.now() - (index + 1) * 3600000).toISOString();
      
      // Third conversation (index 2) will be empty - no messages
      const lastMessage: ChatMessage | undefined = index === 2 ? undefined : {
        id: `msg_${conversationId}_last`,
        conversationId,
        senderId: user.id,
        senderName: user.name,
        senderAvatar: user.avatar,
        text: index === 0 
          ? 'Hey! Did you catch anything today?' 
          : 'That Pomfret looks amazing! Where did you catch it?',
        timestamp: lastMessageTime,
        isRead: index === 0,
        isOffline: false
      };

      this.conversations.set(conversationId, {
        id: conversationId,
        otherUser: user,
        lastMessage,
        unreadCount: index === 0 ? 0 : (index === 2 ? 0 : index),
        updatedAt: lastMessageTime
      });

      // Initialize messages for first two conversations only
      if (index === 0 && !cachedMessages) {
        const chatMessages: ChatMessage[] = [
          {
            id: 'msg1',
            conversationId,
            senderId: user.id,
            senderName: user.name,
            senderAvatar: user.avatar,
            text: 'Hey! How are you doing?',
            timestamp: new Date(Date.now() - 7200000).toISOString(),
            isRead: true
          },
          {
            id: 'msg2',
            conversationId,
            senderId: currentUser.id,
            senderName: currentUser.name,
            senderAvatar: currentUser.avatar || '',
            text: 'Hi Raj! I\'m good, just got back from a great catch!',
            timestamp: new Date(Date.now() - 6800000).toISOString(),
            isRead: true
          },
          {
            id: 'msg3',
            conversationId,
            senderId: user.id,
            senderName: user.name,
            senderAvatar: user.avatar,
            text: 'That\'s awesome! What did you catch?',
            timestamp: new Date(Date.now() - 6400000).toISOString(),
            isRead: true
          },
          {
            id: 'msg4',
            conversationId,
            senderId: currentUser.id,
            senderName: currentUser.name,
            senderAvatar: currentUser.avatar || '',
            text: 'A beautiful Kingfish! The AI identified it with 95% confidence.',
            timestamp: new Date(Date.now() - 6000000).toISOString(),
            isRead: true
          },
          lastMessage!
        ];
        this.messages.set(conversationId, chatMessages);
      } else if (index === 1 && !cachedMessages) {
        const chatMessages: ChatMessage[] = [
          {
            id: 'msg5',
            conversationId,
            senderId: user.id,
            senderName: user.name,
            senderAvatar: user.avatar,
            text: 'Hi! I saw your recent catch post. That Pomfret looks amazing!',
            timestamp: new Date(Date.now() - 5400000).toISOString(),
            isRead: true
          },
          {
            id: 'msg6',
            conversationId,
            senderId: currentUser.id,
            senderName: currentUser.name,
            senderAvatar: currentUser.avatar || '',
            text: 'Thanks! It was a great morning catch at Marina Beach.',
            timestamp: new Date(Date.now() - 5000000).toISOString(),
            isRead: true
          },
          lastMessage!
        ];
        this.messages.set(conversationId, chatMessages);
      }
    });
    
    this.saveCachedMessages();
  }

  getConversations(): Conversation[] {
    return Array.from(this.conversations.values())
      .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
  }

  getMessages(conversationId: string): ChatMessage[] {
    return this.messages.get(conversationId) || [];
  }

  sendMessage(conversationId: string, text: string): ChatMessage {
    const currentUser = authService.getState().user;
    if (!currentUser) throw new Error('User not authenticated');

    const conversation = this.conversations.get(conversationId);
    if (!conversation) throw new Error('Conversation not found');

    const newMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      conversationId,
      senderId: currentUser.id,
      senderName: currentUser.name,
      senderAvatar: currentUser.avatar || '',
      text,
      timestamp: new Date().toISOString(),
      isRead: false,
      isOffline: !navigator.onLine
    };

    // Add message to conversation
    const existingMessages = this.messages.get(conversationId) || [];
    this.messages.set(conversationId, [...existingMessages, newMessage]);

    // Update conversation
    conversation.lastMessage = newMessage;
    conversation.updatedAt = newMessage.timestamp;

    this.saveCachedMessages();
    this.notifyListeners();
    return newMessage;
  }

  private saveCachedMessages() {
    try {
      const messagesObj = Object.fromEntries(this.messages);
      localStorage.setItem('fishnet_chat_messages', JSON.stringify(messagesObj));
    } catch (e) {
      console.warn('Failed to cache messages');
    }
  }

  markAsRead(conversationId: string) {
    const conversation = this.conversations.get(conversationId);
    if (conversation) {
      conversation.unreadCount = 0;
      this.notifyListeners();
    }
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener());
  }
}

export const chatService = new ChatService();
