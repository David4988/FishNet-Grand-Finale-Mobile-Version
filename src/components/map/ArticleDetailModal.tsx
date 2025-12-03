import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { X, ExternalLink } from 'lucide-react';
import { type NewsArticle } from '@/services/news';

interface ArticleDetailModalProps {
  article: NewsArticle | null;
  onOpenChange: (open: boolean) => void;
}

export const ArticleDetailModal = ({ article, onOpenChange }: ArticleDetailModalProps) => {
  if (!article) {
    return null;
  }

  const handleOpenArticle = () => {
    window.open(article.link, '_blank', 'noopener,noreferrer');
  };

  return (
    <Dialog open={!!article} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg w-full p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-4 pb-3 border-b flex flex-row items-center justify-between">
          <DialogTitle className="text-lg font-bold truncate">{article.source}</DialogTitle>
        </DialogHeader>
        <div className="p-6 space-y-4">
            <h2 className="text-2xl font-bold">{article.title}</h2>
            {article.image && (
                <img
                    src={article.image}
                    alt={article.title}
                    className="w-full h-auto object-cover rounded-lg"
                    onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                    }}
                />
            )}
            <p className="text-muted-foreground">{article.description}</p>
            <Button onClick={handleOpenArticle} className="w-full">
                <ExternalLink className="mr-2 h-4 w-4" />
                Read Full Article
            </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};