import { useState, useEffect } from 'react';
// Hardcoded multilingual labels without touching locale files
import { motion, AnimatePresence } from 'framer-motion';
import { Wifi, WifiOff, RefreshCw, Users, MessageSquare, Plus, Heart, MessageCircle, Share2 } from 'lucide-react';

// CORE COMPONENT IMPORTS
import { PostCard } from '@/components/social/PostCard';
import { MessagesDialog } from '@/components/social/MessagesDialog';
import { NewPostDialog } from '@/components/social/NewPostDialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

// CORE SERVICE IMPORTS
import { socialService, SocialPost, User } from '@/services/social';
import { authService } from '@/services/auth';
import { syncService, SyncStatus } from '@/services/sync';

// CORE HOOKS AND UTILS
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';


// --- ANIMATION VARIANTS ---
const feedContainerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const postCardVariants = {
  hidden: { opacity: 0, y: 30, scale: 0.98 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.4, ease: 'easeOut' } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.2 } },
};


export default function FeedPage() {
  const getLang = () => (typeof window !== 'undefined' ? localStorage.getItem('fishnet_language') || 'en' : 'en');
  const lang = getLang();
  const dict: Record<string, Record<string, string>> = {
    en: {
      app_name: 'Fish Net',
      feed_title: 'Community Feed',
      liked_offline: 'Liked offline',
      liked_offline_desc: "Your like will sync when you're back online",
      sample_comment_1: 'Amazing catch! 🎣',
      sample_comment_2: 'What a beauty!',
      sample_comment_3: 'Great photo quality!',
      sample_comment_4: 'This species is rare!',
      sample_comment_5: 'Perfect timing! 🌊',
      comment_added: 'Comment added!',
      comment_posted: 'Comment posted successfully',
      comment_saved_offline: 'Comment saved offline',
      error: 'Error',
      comment_failed: 'Failed to add comment',
      share_title_prefix: 'Check out this',
      link_copied: 'Link copied!',
      share_desc: 'Share this awesome catch with friends.',
      refreshed: 'Feed refreshed',
      latest_loaded: 'Latest posts loaded',
      showing_cached: 'Showing cached posts',
      refresh_failed: 'Refresh failed',
      could_not_refresh: 'Could not refresh feed',
      online: 'Online',
      offline: 'Offline',
      new_post: 'New Post',
      syncing: 'Syncing your data...',
      empty_title: 'The Ocean is Quiet',
      empty_message: 'Be the first to share a catch with the community!',
      create_first_post: 'Create First Post',
    },
    ta: {
      app_name: 'மீன் வலை',
      feed_title: 'சமூகப் பதிவுகள்',
      liked_offline: 'ஆஃப்லைனில் விருப்பம்',
      liked_offline_desc: 'நீங்கள் ஆன்லைனில் வந்ததும் ஒத்திசைக்கப்படும்',
      sample_comment_1: 'அற்புதமான பிடி! 🎣',
      sample_comment_2: 'எவ்வளவு அழகானது!',
      sample_comment_3: 'அற்புதமான புகைப்படம்!',
      sample_comment_4: 'இந்த இனங்கள் அரிது!',
      sample_comment_5: 'சரியான நேரம்! 🌊',
      comment_added: 'கருத்து சேர்க்கப்பட்டது!',
      comment_posted: 'கருத்து வெற்றிகரமாக பதியப்பட்டது',
      comment_saved_offline: 'கருத்து ஆஃப்லைனில் சேமிக்கப்பட்டது',
      error: 'பிழை',
      comment_failed: 'கருத்து சேர்க்க முடியவில்லை',
      share_title_prefix: 'இதைக் பாருங்கள்',
      link_copied: 'இணைப்பு நகலெடுக்கப்பட்டது!',
      share_desc: 'இந்த அருமையான பிடியை நண்பர்களுடன் பகிருங்கள்.',
      refreshed: 'பதிவுகள் புதுப்பிக்கப்பட்டது',
      latest_loaded: 'சமீபத்திய பதிவுகள் ஏற்றப்பட்டது',
      showing_cached: 'கேச் செய்யப்பட்ட பதிவுகள் காட்டப்படுகின்றன',
      refresh_failed: 'புதுப்பித்தல் தோல்வியடைந்தது',
      could_not_refresh: 'பதிவுகளை புதுப்பிக்க முடியவில்லை',
      online: 'ஆன்லைன்',
      offline: 'ஆஃப்லைன்',
      new_post: 'புதிய பதிவு',
      syncing: 'தரவை ஒத்திசைக்கிறது...',
      empty_title: 'கடல் அமைதியாக உள்ளது',
      empty_message: 'முதலில் பிடியை சமூகத்துடன் பகிருங்கள்!',
      create_first_post: 'முதல் பதிவை உருவாக்கவும்',
    },
    hi: {
      app_name: 'मछली जाल',
      feed_title: 'समुदाय फ़ीड',
      liked_offline: 'ऑफ़लाइन पसंद किया',
      liked_offline_desc: 'आप ऑनलाइन होने पर सिंक होगा',
      sample_comment_1: 'कमाल की पकड़! 🎣',
      sample_comment_2: 'वाह क्या बात है!',
      sample_comment_3: 'बहुत बढ़िया फोटो!',
      sample_comment_4: 'यह प्रजाति दुर्लभ है!',
      sample_comment_5: 'सही समय! 🌊',
      comment_added: 'टिप्पणी जोड़ी गई!',
      comment_posted: 'टिप्पणी सफलतापूर्वक पोस्ट हुई',
      comment_saved_offline: 'टिप्पणी ऑफ़लाइन सहेजी गई',
      error: 'त्रुटि',
      comment_failed: 'टिप्पणी जोड़ने में विफल',
      share_title_prefix: 'देखें यह',
      link_copied: 'लिंक कॉपी किया गया!',
      share_desc: 'इस शानदार पकड़ को साझा करें।',
      refreshed: 'फ़ीड ताज़ा किया गया',
      latest_loaded: 'नवीनतम पोस्ट लोडेड',
      showing_cached: 'कैश किए गए पोस्ट दिखाए जा रहे हैं',
      refresh_failed: 'ताज़ा करना विफल रहा',
      could_not_refresh: 'फ़ीड को ताज़ा नहीं कर सके',
      online: 'ऑनलाइन',
      offline: 'ऑफ़लाइन',
      new_post: 'नई पोस्ट',
      syncing: 'आपका डेटा सिंक हो रहा है...',
      empty_title: 'समुद्र शांत है',
      empty_message: 'समुदाय के साथ पहली पकड़ साझा करें!',
      create_first_post: 'पहली पोस्ट बनाएँ',
    },
  };
  const tr = (key: string, fallback?: string) => (dict[lang]?.[key] ?? fallback ?? dict.en[key] ?? key);
  const [posts, setPosts] = useState<SocialPost[]>([]);
  const [currentUser, setCurrentUser] = useState<User | null>(authService.getState().user);
  const [syncStatus, setSyncStatus] = useState<SyncStatus>(syncService.getStatus());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { toast } = useToast();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [fabOpen, setFabOpen] = useState(false);

  useEffect(() => {
    loadPosts(); // Initial load

    const unsubscribeAuth = authService.subscribe((state) => setCurrentUser(state.user));
    const unsubscribeSync = syncService.subscribe((status) => {
      setSyncStatus(status);
      if (!status.isSyncing && status.isOnline) {
        loadPosts();
      }
    });
    // Subscribe to the social service for real-time post updates
    const unsubscribeSocial = socialService.subscribe(loadPosts);

    return () => {
      unsubscribeAuth();
      unsubscribeSync();
      unsubscribeSocial();
    };
  }, []);

  const loadPosts = () => {
    const feedPosts = socialService.getFeedPosts();
    setPosts(feedPosts);
  };

  const handleLike = async (postId: string) => {
    if (!currentUser) return;
    // The socialService now handles the state update and notifies listeners,
    // so we don't need to manually setPosts here anymore.
    await socialService.toggleLike(postId, currentUser.id);
    if (!syncStatus.isOnline) {
      toast({ title: tr('liked_offline'), description: tr('liked_offline_desc') });
    }
  };

  const handleComment = async (postId: string) => {
    if (!currentUser) return;
    try {
      const sampleComments = [
        tr('sample_comment_1'),
        tr('sample_comment_2'),
        tr('sample_comment_3'),
        tr('sample_comment_4'),
        tr('sample_comment_5'),
      ];
      const randomComment = sampleComments[Math.floor(Math.random() * sampleComments.length)];
      await socialService.addComment(postId, currentUser.id, currentUser, randomComment);
      toast({ title: tr('comment_added'), description: syncStatus.isOnline ? tr('comment_posted') : tr('comment_saved_offline') });
    } catch (error) {
      toast({ title: tr('error'), description: tr('comment_failed'), variant: "destructive" });
    }
  };

  const handleShare = (postId: string) => {
    const post = posts.find(p => p.id === postId);
    if (!post) return;
    if (navigator.share) {
      navigator.share({ 
        title: `${tr('share_title_prefix')} ${post.species} catch on ${tr('app_name')}!`, 
        text: post.caption, 
        url: window.location.href 
      });
    } else {
      navigator.clipboard.writeText(window.location.href);
      toast({ title: tr('link_copied'), description: tr('share_desc') });
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      if (syncStatus.isOnline) await syncService.forcSync();
      loadPosts(); // Manually reload posts on refresh
      toast({ title: tr('refreshed'), description: syncStatus.isOnline ? tr('latest_loaded') : tr('showing_cached') });
    } catch (error) {
      toast({ title: tr('refresh_failed'), description: tr('could_not_refresh'), variant: "destructive" });
    } finally {
      setIsRefreshing(false);
    }
  };

  const isPostLiked = (postId: string) => currentUser ? socialService.isPostLiked(postId, currentUser.id) : false;

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <div className="absolute inset-0 bg-grid-sky-400/[0.05]" />
      
      <header className="sticky top-0 z-40 bg-slate-900/60 backdrop-blur-xl border-b border-sky-400/20">
        <div className="flex items-center justify-between p-4 h-20">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-2xl font-bold text-sky-400 tracking-wider">{tr('app_name')}</h1>
            <p>{tr('feed_title')}</p>
            </div>
            <Badge className={cn("hidden sm:inline-flex items-center text-xs", syncStatus.isOnline ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" : "bg-amber-500/20 text-amber-300 border-amber-500/30")}>
              {syncStatus.isOnline ? <><Wifi className="h-3 w-3 mr-1.5" /> {tr('online')}</> : <><WifiOff className="h-3 w-3 mr-1.5" /> {tr('offline')}</>}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden sm:block">
              <NewPostDialog onPostCreated={loadPosts}>
                <Button className="bg-sky-500 hover:bg-sky-600 text-white font-bold rounded-full shadow-lg shadow-sky-500/30 transition-all duration-300">
                  <Plus className="h-5 w-5 mr-2" /> {tr('new_post')}
                </Button>
              </NewPostDialog>
            </div>
            <MessagesDialog>
                <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-sky-500/20">
                    <Users className="h-5 w-5" />
                </Button>
            </MessagesDialog>
            {/* --- EDIT: Hidden on mobile, visible on sm screens and up --- */}
            <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={isRefreshing} className="hidden sm:flex text-gray-400 hover:text-white hover:bg-sky-500/20">
              <RefreshCw className={cn("h-5 w-5", isRefreshing && "animate-spin")} />
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-md p-4 pb-32">
        {syncStatus.isSyncing && (
          <div className="bg-sky-500/20 p-3 rounded-lg text-center text-sm text-sky-300 flex items-center justify-center gap-2 mb-4">
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>{tr('syncing')}</span>
          </div>
        )}
        <AnimatePresence mode="wait">
          {posts.length > 0 ? (
            <motion.div
              key="posts-list"
              initial="hidden"
              animate="visible"
              exit={postCardVariants.exit}
              variants={feedContainerVariants}
              className="space-y-6"
            >
              {posts.map((post) => (<PostCard key={post.id} post={post} onLike={handleLike} onComment={handleComment} onShare={handleShare} isLiked={isPostLiked(post.id)} />))}
            </motion.div>
          ) : (
            <motion.div
              key="empty-state"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={postCardVariants.exit}
              className="text-center py-20"
            >
              <div className="w-20 h-20 bg-sky-500/10 border-2 border-sky-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                <Users className="h-10 w-10 text-sky-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">{tr('empty_title')}</h3>
              <p className="text-gray-400 mb-6">{tr('empty_message')}</p>
              <NewPostDialog onPostCreated={loadPosts}>
                <Button size="lg" className="bg-sky-500 hover:bg-sky-600 text-white font-bold rounded-full shadow-lg shadow-sky-500/30 transition-all duration-300"><Plus className="h-5 w-5 mr-2" /> {tr('create_first_post')}</Button>
              </NewPostDialog>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <div className="sm:hidden fixed bottom-24 right-4 z-40">
         <NewPostDialog onPostCreated={loadPosts}>
             <motion.div
                 whileHover={{ scale: 1.1 }}
                 whileTap={{ scale: 0.9 }}
              >
                <Button
                  className="rounded-full h-16 w-16 bg-sky-500 hover:bg-sky-600 shadow-xl shadow-sky-500/40 text-white"
                >
                    <Plus size={28} />
                </Button>
              </motion.div>
        </NewPostDialog>
      </div>

    </div>
  );
}

