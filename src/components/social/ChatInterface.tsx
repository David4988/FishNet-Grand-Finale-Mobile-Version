import { useState, useEffect, useRef, useCallback } from 'react';
import { Send, ArrowLeft } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { chatService, Conversation, ChatMessage } from '@/services/chat';
import { authService } from '@/services/auth';
import { formatDistanceToNow } from 'date-fns';
import { cn } from '@/lib/utils';

export const ChatInterface = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [currentUser, setCurrentUser] = useState(authService.getState().user);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isMobileView, setIsMobileView] = useState(false);
  const selectedConversationRef = useRef<Conversation | null>(null);
  const hasAutoSelectedRef = useRef(false);

  const loadConversations = useCallback(() => {
    const convs = chatService.getConversations();
    setConversations(convs);
  }, []);

  const loadMessages = useCallback((conversationId: string) => {
    const msgs = chatService.getMessages(conversationId);
    setMessages(msgs);
  }, []);

  useEffect(() => {
    // Initialize mobile view check
    if (typeof window !== 'undefined') {
      setIsMobileView(window.innerWidth < 768);
    }

    loadConversations();
    
    const unsubscribeChat = chatService.subscribe(() => {
      loadConversations();
      if (selectedConversationRef.current) {
        loadMessages(selectedConversationRef.current.id);
      }
    });

    const unsubscribeAuth = authService.subscribe((state) => {
      setCurrentUser(state.user);
      if (state.user) {
        // Reinitialize conversations when user logs in
        chatService.initialize();
        loadConversations();
        hasAutoSelectedRef.current = false;
      }
    });

    const handleResize = () => {
      setIsMobileView(window.innerWidth < 768);
    };
    
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', handleResize);
    }

    return () => {
      unsubscribeChat();
      unsubscribeAuth();
      if (typeof window !== 'undefined') {
        window.removeEventListener('resize', handleResize);
      }
    };
  }, [loadConversations, loadMessages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-select first conversation on desktop (only once)
  useEffect(() => {
    if (!isMobileView && conversations.length > 0 && !selectedConversationRef.current && !hasAutoSelectedRef.current) {
      handleSelectConversation(conversations[0]);
      hasAutoSelectedRef.current = true;
    }
  }, [conversations, isMobileView]);

  const handleSelectConversation = (conversation: Conversation) => {
    setSelectedConversation(conversation);
    selectedConversationRef.current = conversation;
    loadMessages(conversation.id);
    chatService.markAsRead(conversation.id);
  };

  const handleSendMessage = () => {
    if (!newMessage.trim() || !selectedConversation || !currentUser) return;

    chatService.sendMessage(selectedConversation.id, newMessage);
    setNewMessage('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const formatTime = (timestamp: string) => {
    return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
  };

  // Mobile view: show either conversation list or selected chat
  if (isMobileView && selectedConversation) {
    return (
      <div className="h-[calc(100dvh-180px)] flex flex-col bg-background">
        {/* Chat Header */}
        <div className="flex items-center gap-3 p-4 border-b border-border bg-card">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedConversation(null)}
            data-testid="button-back-to-conversations"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <Avatar className="h-10 w-10">
            <AvatarImage src={selectedConversation.otherUser.avatar} alt={selectedConversation.otherUser.name} />
            <AvatarFallback>
              {selectedConversation.otherUser.name.slice(0, 2).toUpperCase()}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1">
            <h3 className="font-semibold text-foreground">{selectedConversation.otherUser.name}</h3>
            <p className="text-xs text-muted-foreground">{selectedConversation.otherUser.bio}</p>
          </div>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages.map((message) => {
              const isOwn = message.senderId === currentUser?.id;
              return (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-2",
                    isOwn ? "justify-end" : "justify-start"
                  )}
                  data-testid={`message-${message.id}`}
                >
                  {!isOwn && (
                    <Avatar className="h-8 w-8">
                      <AvatarImage src={message.senderAvatar} alt={message.senderName} />
                      <AvatarFallback className="text-xs">
                        {message.senderName.slice(0, 2).toUpperCase()}
                      </AvatarFallback>
                    </Avatar>
                  )}
                  <div className={cn(
                    "max-w-[75%] rounded-2xl px-4 py-2",
                    isOwn 
                      ? "bg-gradient-primary text-white" 
                      : "bg-muted text-foreground"
                  )}>
                    <p className="text-sm">{message.text}</p>
                    <p className={cn(
                      "text-xs mt-1",
                      isOwn ? "text-white/70" : "text-muted-foreground"
                    )}>
                      {formatTime(message.timestamp)}
                    </p>
                  </div>
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Message Input */}
        <div className="p-4 border-t border-border bg-card">
          <div className="flex gap-2">
            <Input
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type a message..."
              className="flex-1"
              data-testid="input-message"
            />
            <Button 
              onClick={handleSendMessage}
              disabled={!newMessage.trim()}
              className="bg-gradient-primary hover:opacity-90 text-white"
              data-testid="button-send-message"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100dvh-180px)] flex bg-background">
      {/* Conversations List */}
      <div className={cn(
        "border-r border-border bg-card",
        isMobileView ? "w-full" : "w-80"
      )}>
        <ScrollArea className="h-full">
          <div className="p-2">
            {conversations.length > 0 ? (
              conversations.map((conversation) => (
                <button
                  key={conversation.id}
                  onClick={() => handleSelectConversation(conversation)}
                  className={cn(
                    "w-full p-3 flex items-start gap-3 rounded-lg transition-colors",
                    "hover:bg-muted",
                    selectedConversation?.id === conversation.id && "bg-muted"
                  )}
                  data-testid={`conversation-${conversation.id}`}
                >
                  <Avatar className="h-12 w-12">
                    <AvatarImage src={conversation.otherUser.avatar} alt={conversation.otherUser.name} />
                    <AvatarFallback>
                      {conversation.otherUser.name.slice(0, 2).toUpperCase()}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0 text-left">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="font-semibold text-foreground truncate">
                        {conversation.otherUser.name}
                      </h4>
                      {conversation.unreadCount > 0 && (
                        <Badge className="bg-primary text-primary-foreground text-xs">
                          {conversation.unreadCount}
                        </Badge>
                      )}
                    </div>
                    {conversation.lastMessage && (
                      <>
                        <p className="text-sm text-muted-foreground truncate">
                          {conversation.lastMessage.text}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {formatTime(conversation.lastMessage.timestamp)}
                        </p>
                      </>
                    )}
                  </div>
                </button>
              ))
            ) : (
              <div className="text-center py-12" data-testid="empty-conversations">
                <p className="text-muted-foreground">No conversations yet</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Desktop: Chat Area */}
      {!isMobileView && (
        <div className="flex-1 flex flex-col">
          {selectedConversation ? (
            <>
              {/* Chat Header */}
              <div className="flex items-center gap-3 p-4 border-b border-border bg-card">
                <Avatar className="h-10 w-10">
                  <AvatarImage src={selectedConversation.otherUser.avatar} alt={selectedConversation.otherUser.name} />
                  <AvatarFallback>
                    {selectedConversation.otherUser.name.slice(0, 2).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <h3 className="font-semibold text-foreground">{selectedConversation.otherUser.name}</h3>
                  <p className="text-xs text-muted-foreground">{selectedConversation.otherUser.bio}</p>
                </div>
              </div>

              {/* Messages */}
              <ScrollArea className="flex-1 p-4">
                <div className="space-y-4">
                  {messages.map((message) => {
                    const isOwn = message.senderId === currentUser?.id;
                    return (
                      <div
                        key={message.id}
                        className={cn(
                          "flex gap-2",
                          isOwn ? "justify-end" : "justify-start"
                        )}
                        data-testid={`message-${message.id}`}
                      >
                        {!isOwn && (
                          <Avatar className="h-8 w-8">
                            <AvatarImage src={message.senderAvatar} alt={message.senderName} />
                            <AvatarFallback className="text-xs">
                              {message.senderName.slice(0, 2).toUpperCase()}
                            </AvatarFallback>
                          </Avatar>
                        )}
                        <div className={cn(
                          "max-w-[75%] rounded-2xl px-4 py-2",
                          isOwn 
                            ? "bg-gradient-primary text-white" 
                            : "bg-muted text-foreground"
                        )}>
                          <p className="text-sm">{message.text}</p>
                          <p className={cn(
                            "text-xs mt-1",
                            isOwn ? "text-white/70" : "text-muted-foreground"
                          )}>
                            {formatTime(message.timestamp)}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>

              {/* Message Input */}
              <div className="p-4 border-t border-border bg-card">
                <div className="flex gap-2">
                  <Input
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type a message..."
                    className="flex-1"
                    data-testid="input-message"
                  />
                  <Button 
                    onClick={handleSendMessage}
                    disabled={!newMessage.trim()}
                    className="bg-gradient-primary hover:opacity-90 text-white"
                    data-testid="button-send-message"
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center" data-testid="empty-chat-selection">
              <div className="text-center">
                <p className="text-lg font-semibold text-foreground mb-2">Select a conversation</p>
                <p className="text-muted-foreground">Choose a conversation from the list to start chatting</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
