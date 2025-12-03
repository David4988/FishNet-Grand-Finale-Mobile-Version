import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Newspaper, ExternalLink, Loader2, RefreshCw, Clock } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import { getLatestNews, refreshNews, type NewsArticle } from '@/services/news';
import { ArticleDetailModal } from './ArticleDetailModal';

export const NewsModule = () => {
  const { t, i18n } = useTranslation();
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedArticle, setSelectedArticle] = useState<NewsArticle | null>(null);

  useEffect(() => {
    loadNews();

    const intervalId = setInterval(() => {
      handleRefresh();
    }, 30 * 60 * 1000); // 30 minutes

    return () => {
      clearInterval(intervalId);
    };
  }, [i18n.language]);

  const loadNews = async () => {
    setLoading(true);
    setError(null);
    try {
      const storedNews = getLatestNews(i18n.language);
      if (storedNews.length > 0) {
        setArticles(storedNews);
      } else {
        const freshNews = await refreshNews(i18n.language);
        setArticles(freshNews);
      }
    } catch (err) {
      console.error('Error loading news:', err);
      setError(t('map.news.noNews'));
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      const freshNews = await refreshNews(i18n.language);
      setArticles(freshNews);
    } catch (err) {
      console.error('Error refreshing news:', err);
    } finally {
      setRefreshing(false);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat(i18n.language, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch {
      return dateString;
    }
  };

  const getTimeAgo = (dateString: string) => {
    try {
      const date = new Date(dateString);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      const diffDays = Math.floor(diffHours / 24);

      if (diffDays > 0) {
        return `${diffDays}d ago`;
      } else if (diffHours > 0) {
        return `${diffHours}h ago`;
      } else {
        const diffMins = Math.floor(diffMs / (1000 * 60));
        return `${diffMins}m ago`;
      }
    } catch {
      return '';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">{t('map.news.loading')}</p>
        </div>
      </div>
    );
  }

  if (error || articles.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Newspaper className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
          <p className="text-lg font-medium mb-4">{error || t('map.news.noNews')}</p>
          <Button onClick={handleRefresh} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            {t('common.refresh', { defaultValue: 'Refresh' })}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-6 py-4 border-b bg-background/95 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <Newspaper className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-bold">{t('map.news.title')}</h2>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          {t('common.refresh', { defaultValue: 'Refresh' })}
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-6 space-y-4">
          {articles.map((article, index) => (
            <motion.div
              key={article.link || index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card className="overflow-hidden hover:shadow-lg transition-all duration-300 border-l-4 border-l-primary/50">
                <div
                  onClick={() => setSelectedArticle(article)}
                  className="block cursor-pointer"
                >
                  <div className="flex flex-col sm:flex-row">
                    {article.image && (
                      <div className="sm:w-48 h-32 sm:h-auto flex-shrink-0 bg-muted">
                        <img
                          src={article.image}
                          alt={article.title}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                      </div>
                    )}
                    <div className="flex-1 p-4">
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <h3 className="font-semibold text-base sm:text-lg line-clamp-2 hover:text-primary transition-colors">
                          {article.title}
                        </h3>
                        <ExternalLink className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                      </div>

                      {article.description && (
                        <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                          {article.description}
                        </p>
                      )}

                      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                        <span className="font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded">
                          {article.source}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {getTimeAgo(article.pubDate)}
                        </span>
                        <span className="hidden sm:inline">
                          {formatDate(article.pubDate)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </ScrollArea>
      <ArticleDetailModal article={selectedArticle} onOpenChange={() => setSelectedArticle(null)} />
    </div>
  );
};