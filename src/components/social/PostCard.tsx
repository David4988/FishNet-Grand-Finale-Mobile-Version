import { useState } from 'react';
// Hardcoded multilingual labels without touching locale files
import { Heart, MessageCircle, Share, MapPin, Calendar, Fish } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { SocialPost } from '@/services/social';
import { formatDistanceToNow, type Locale } from 'date-fns';
import { enUS, hi, ta, te, kn, gu, bn } from 'date-fns/locale';
import { cn } from '@/lib/utils';
import { translateSpecies } from '@/utils/translateSpecies';

const localeMap: Record<string, Locale> = {
  en: enUS,
  hi: hi,
  ta: ta,
  te: te,
  kn: kn,
  ml: enUS, // Malayalam - fallback to English (no ml locale in date-fns)
  gu: gu,
  bn: bn,
  pa: enUS, // Punjabi - fallback to English (no pa locale in date-fns)
  mr: enUS, // Marathi - fallback to English (no mr locale in date-fns)
  mwr: enUS, // Marwadi - fallback to English (no mwr locale in date-fns)
  or: enUS,  // Odia - fallback to English (no or locale in date-fns)
};

interface PostCardProps {
  post: SocialPost;
  onLike: (postId: string) => void;
  onComment: (postId: string) => void;
  onShare: (postId: string) => void;
  isLiked: boolean;
  className?: string;
}

export const PostCard = ({ 
  post, 
  onLike, 
  onComment, 
  onShare, 
  isLiked, 
  className 
}: PostCardProps) => {
  const getLang = () => (typeof window !== 'undefined' ? localStorage.getItem('fishnet_language') || 'en' : 'en');
  const lang = getLang();
  const dict: Record<string, Record<string, string>> = {
    en: {
      post_offline: 'Offline',
      post_weight: 'Estimated Weight',
      post_health_score: 'Health Score',
      post_view_all_comments: 'View all comments',
    },
    ta: {
      post_offline: 'ஆஃப்லைன்',
      post_weight: 'மதிப்பிடப்பட்ட எடை',
      post_health_score: 'உடல் நலம் மதிப்பெண்',
      post_view_all_comments: 'அனைத்து கருத்துகளையும் காண்க',
    },
    hi: {
      post_offline: 'ऑफ़लाइन',
      post_weight: 'अनुमानित वजन',
      post_health_score: 'स्वास्थ्य स्कोर',
      post_view_all_comments: 'सभी टिप्पणियाँ देखें',
    },
  };
  const tr = (key: string, fallback?: string) => (dict[lang]?.[key] ?? fallback ?? dict.en[key] ?? key);
  const [imageLoaded, setImageLoaded] = useState(false);

  const formatTimeAgo = (timestamp: string) => {
    const locale = localeMap[lang] || enUS;
    const distance = formatDistanceToNow(new Date(timestamp), { locale, addSuffix: true });
    return distance;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-success';
    if (confidence >= 60) return 'text-warning';
    return 'text-destructive';
  };

  return (
    <Card className={cn(
      "overflow-hidden bg-gradient-to-b from-card to-card/80",
      "border border-border/50",
      "shadow-card hover:shadow-float transition-all duration-300",
      className
    )}>
      {/* Post Header */}
      <div className="flex items-center gap-3 p-4 pb-2">
        <Avatar className="h-10 w-10 ring-2 ring-primary/20">
          <AvatarImage src={post.user.avatar} alt={post.user.name} />
          <AvatarFallback className="bg-primary/10 text-primary">
            {post.user.name.slice(0, 2).toUpperCase()}
          </AvatarFallback>
        </Avatar>
        
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-foreground">{post.user.name}</h3>
            {post.isOffline && (
              <Badge variant="outline" className="text-xs bg-warning/10 text-warning border-warning/20">
                {tr('post_offline')}
              </Badge>
            )}
          </div>
          
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Calendar className="h-3 w-3" />
            <span>{formatTimeAgo(post.timestamp)}</span>
            {post.location && (
              <>
                <span>•</span>
                <MapPin className="h-3 w-3" />
                <span>{post.location.name}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Fish Image */}
      <div className="relative overflow-hidden">
        <img
          src={post.imageData}
          alt={`${post.species} catch`}
          className={cn(
            "w-full aspect-square object-cover transition-all duration-500",
            imageLoaded ? "opacity-100 scale-100" : "opacity-0 scale-105"
          )}
          onLoad={() => setImageLoaded(true)}
        />
        
        {!imageLoaded && (
          <div className="absolute inset-0 bg-muted animate-pulse flex items-center justify-center">
            <Fish className="h-8 w-8 text-muted-foreground animate-bounce" />
          </div>
        )}

        {/* Species Overlay */}
        <div className="absolute top-4 left-4 right-4">
          <Badge className="bg-black/70 text-white backdrop-blur-sm border-white/20">
            <Fish className="h-3 w-3 mr-1" />
            {translateSpecies(post.species, (key: string) => key)}
          </Badge>
        </div>

        {/* Confidence Badge
        <div className="absolute top-4 right-4">
          <Badge 
            variant="secondary" 
            className={cn(
              "bg-blackx/70 backdrop-blur-sm border-white/20",
              getConfidenceColor(post.confidence)
            )}
          >
            {post.confidence}% confident
          </Badge>
        </div> */}
      </div>

      {/* Post Stats */}
      <div className="p-4 space-y-3">
        {/* Fish Details */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-muted-foreground mb-1">{tr('post_weight')}</div>
            <div className="font-semibold text-foreground">{post.estimatedWeight.toFixed(1)} kg</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground mb-1">{tr('post_health_score')}</div>
            <div className="flex items-center gap-2">
              <Progress 
                value={post.healthScore} 
                className="flex-1 h-2"
              />
              <span className="text-sm font-medium text-foreground">{post.healthScore}%</span>
            </div>
          </div>
        </div>

        {/* Caption */}
        {post.caption && (
          <p className="text-foreground leading-relaxed">
            {post.caption}
          </p>
        )}

        {/* Interaction Buttons */}
        <div className="flex items-center justify-between pt-2 border-t border-border/50">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onLike(post.id)}
              className={cn(
                "flex items-center gap-2 transition-all duration-200",
                isLiked 
                  ? "text-like hover:text-like/80" 
                  : "text-muted-foreground hover:text-like"
              )}
            >
              <Heart 
                className={cn(
                  "h-5 w-5 transition-all duration-200",
                  isLiked && "fill-current scale-110"
                )} 
              />
              <span className="font-medium">{post.likes}</span>
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => onComment(post.id)}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground"
            >
              <MessageCircle className="h-5 w-5" />
              <span className="font-medium">{post.comments.length}</span>
            </Button>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={() => onShare(post.id)}
            className="text-muted-foreground hover:text-foreground"
          >
            <Share className="h-5 w-5" />
          </Button>
        </div>

        {/* Comments Preview */}
        {post.comments.length > 0 && (
          <div className="pt-2 space-y-2">
            {post.comments.slice(0, 2).map((comment) => (
              <div key={comment.id} className="flex items-start gap-2">
                <Avatar className="h-6 w-6">
                  <AvatarImage src={comment.user.avatar} alt={comment.user.name} />
                  <AvatarFallback className="text-xs">
                    {comment.user.name.slice(0, 2).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <span className="text-sm">
                    <span className="font-medium text-foreground">{comment.user.name}</span>{' '}
                    <span className="text-muted-foreground">{comment.text}</span>
                  </span>
                </div>
              </div>
            ))}
            
            {post.comments.length > 2 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onComment(post.id)}
                className="text-muted-foreground hover:text-foreground p-0 h-auto"
              >
                {tr('post_view_all_comments')} ({post.comments.length})
              </Button>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};