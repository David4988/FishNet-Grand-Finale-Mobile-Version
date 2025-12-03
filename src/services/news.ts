export interface NewsArticle {
  title: string;
  description: string;
  link: string;
  pubDate: string;
  source: string;
  image: string | null;
}

interface RSSItem {
  title: string;
  description: string;
  link: string;
  pubDate: string;
  author?: string;
  thumbnail?: string;
  enclosure?: {
    link?: string;
    thumbnail?: string;
  };
}

interface RSSResponse {
  status: string;
  items: RSSItem[];
}

const NEWS_STORAGE_KEY_PREFIX = 'fisherman_news'; 
const FETCH_INTERVAL = 30 * 60 * 1000;
const MAX_ARTICLES = 15;

let fetchIntervalId: ReturnType<typeof setInterval> | null = null;

function getStorageKey(language: string): string { 
   return `${NEWS_STORAGE_KEY_PREFIX}_${language}`;
}
function getStoredNews(language: string): NewsArticle[] { 
  try {
    const stored = localStorage.getItem(getStorageKey(language));
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error('Error reading stored news:', error);
  }
  return [];
}

function storeNews(articles: NewsArticle[], language: string): void { 
  try {
    localStorage.setItem(getStorageKey(language), JSON.stringify(articles)); 
  } catch (error) {
    console.error('Error storing news:', error);
  }
}

function isDuplicate(article: NewsArticle, existingArticles: NewsArticle[]): boolean {
  return existingArticles.some(
    (existing) => existing.link === article.link || existing.title === article.title
  );
}

async function fetchNewsFromAPI(language: string): Promise<NewsArticle[]> {
  const rssUrl = `https://news.google.com/rss/search?q=fishermen&hl=${language}&gl=IN&ceid=IN:${language}`;
  const apiUrl = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssUrl)}`;

  try {
    const response = await fetch(apiUrl);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: RSSResponse = await response.json();

    if (data.status !== 'ok' || !data.items) {
      throw new Error('Invalid RSS response');
    }

    return data.items.map((item: RSSItem) => ({
      title: item.title || '',
      description: item.description ? item.description.replace(/<[^>]*>/g, '').substring(0, 200) : '',
      link: item.link || '',
      pubDate: item.pubDate || '',
      source: item.author || extractSourceFromTitle(item.title) || 'Unknown Source',
      image: item.thumbnail || item.enclosure?.thumbnail || item.enclosure?.link || null,
    }));
  } catch (error) {
    console.error('Error fetching news:', error);
    return [];
  }
}

function extractSourceFromTitle(title: string): string {
  const match = title?.match(/- ([^-]+)$/);
  return match ? match[1].trim() : '';
}

export async function refreshNews(language: string): Promise<NewsArticle[]> {
  const existingArticles = getStoredNews(language); 
  const newArticles = await fetchNewsFromAPI(language);

  const uniqueNewArticles = newArticles.filter(
    (article) => !isDuplicate(article, existingArticles)
  );

  const combinedArticles = [...uniqueNewArticles, ...existingArticles]
    .sort((a, b) => new Date(b.pubDate).getTime() - new Date(a.pubDate).getTime())
    .slice(0, MAX_ARTICLES);

  storeNews(combinedArticles, language);
  return combinedArticles;
}

export function getLatestNews(language: string): NewsArticle[] {
  return getStoredNews(language); 
}

export function startNewsFetcher(language: string): void {
  if (fetchIntervalId) {
    clearInterval(fetchIntervalId);
  }

  refreshNews(language).catch(console.error);

  fetchIntervalId = setInterval(() => {
    refreshNews(language).catch(console.error);
  }, FETCH_INTERVAL);
}

export function stopNewsFetcher(): void {
  if (fetchIntervalId) {
    clearInterval(fetchIntervalId);
    fetchIntervalId = null;
  }
}
