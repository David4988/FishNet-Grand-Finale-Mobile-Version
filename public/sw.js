// Fish Net Service Worker - Full Offline PWA Support
const CACHE_NAME = 'fishnet-v1.2.1';
const STATIC_CACHE = 'fishnet-static-v1.2.1';
const DYNAMIC_CACHE = 'fishnet-dynamic-v1.2.1';

// Files to cache for offline functionality
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/favicon.ico',
  '/placeholder.svg',
  '/news/local-news.json',
  '/models/species.json',
  '/models/model.tflite',
  // WebLLM runtime and models (add these as they become available)
  '/web-llm/webllm.mjs',
  '/web-llm/webllm.js',
  '/js/webllm.mjs',
  '/js/webllm.js',
  // Core app routes
  '/analyze',
  '/map', 
  '/history',
  '/profile',
  '/feed'
];// Network-first resources (try network, fallback to cache)
const NETWORK_FIRST = [
  '/api/',
  'https://api.',
  'https://sql.js.org/'
];

// Cache-first resources (serve from cache, update in background)
const CACHE_FIRST = [
  '.js',
  '.css',
  '.woff2',
  '.woff',
  '.ttf',
  '.png',
  '.jpg',
  '.jpeg',
  '.svg',
  '.webp',
  '/models/'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('[SW] Installing Fish Net Service Worker');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[SW] Static assets cached successfully');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('[SW] Failed to cache static assets:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[SW] Activating Fish Net Service Worker');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE && 
                cacheName !== DYNAMIC_CACHE && 
                cacheName !== CACHE_NAME) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[SW] Service Worker activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - handle all network requests
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip chrome-extension and other non-http requests
  if (!url.protocol.startsWith('http')) {
    return;
  }

  event.respondWith(handleFetch(request));
});

async function handleFetch(request) {
  const url = new URL(request.url);
  
  try {
    // Network-first strategy for API calls
    if (NETWORK_FIRST.some(pattern => url.href.includes(pattern))) {
      return await networkFirstStrategy(request);
    }
    
    // Cache-first strategy for static assets
    if (CACHE_FIRST.some(pattern => url.href.includes(pattern))) {
      return await cacheFirstStrategy(request);
    }
    
    // Special handling for navigation requests (HTML pages)
    if (request.mode === 'navigate') {
      return await navigationStrategy(request);
    }
    
    // Default: try network first, fallback to cache
    return await networkFirstStrategy(request);
    
  } catch (error) {
    console.error('[SW] Fetch failed:', error);
    return await getOfflineFallback(request);
  }
}

// Network-first strategy
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Cache successful responses
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request.clone(), networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    // Network failed, try cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

// Cache-first strategy
async function cacheFirstStrategy(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    // Update cache in background
    fetch(request)
      .then(networkResponse => {
        if (networkResponse.ok) {
          caches.open(DYNAMIC_CACHE)
            .then(cache => cache.put(request, networkResponse));
        }
      })
      .catch(() => {
        // Network update failed, but we have cache
      });
    
    return cachedResponse;
  }
  
  // Not in cache, fetch from network
  const networkResponse = await fetch(request);
  
  if (networkResponse.ok) {
    const cache = await caches.open(DYNAMIC_CACHE);
    cache.put(request.clone(), networkResponse.clone());
  }
  
  return networkResponse;
}

// Navigation strategy for SPA routing
async function navigationStrategy(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    return networkResponse;
  } catch (error) {
    // Network failed, serve cached index.html for SPA routing
    const cachedIndex = await caches.match('/index.html');
    if (cachedIndex) {
      return cachedIndex;
    }
    
    // Last resort: offline page
    return getOfflineFallback(request);
  }
}

// Offline fallback responses
async function getOfflineFallback(request) {
  const url = new URL(request.url);
  
  // Try to find any cached version first
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // Navigation requests - serve index.html
  if (request.mode === 'navigate') {
    const indexCache = await caches.match('/index.html');
    if (indexCache) {
      return indexCache;
    }
  }
  
  // Image requests - serve placeholder
  if (request.destination === 'image') {
    const placeholder = await caches.match('/placeholder.svg');
    if (placeholder) {
      return placeholder;
    }
  }
  
  // Create a simple offline response
  return new Response(
    JSON.stringify({
      error: 'Offline',
      message: 'This resource is not available offline',
      timestamp: new Date().toISOString()
    }),
    {
      status: 503,
      statusText: 'Service Unavailable',
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache'
      }
    }
  );
}

// Background sync for offline actions
self.addEventListener('sync', event => {
  console.log('[SW] Background sync:', event.tag);
  
  if (event.tag === 'sync-catches') {
    event.waitUntil(syncCatches());
  }
});

// Sync offline catches when connection is restored
async function syncCatches() {
  try {
    console.log('[SW] Syncing offline catches...');
    
    // This would typically communicate with your app's sync mechanism
    // For now, we'll just log that sync is happening
    
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage({
        type: 'SYNC_CATCHES',
        timestamp: Date.now()
      });
    });
    
    console.log('[SW] Sync completed');
  } catch (error) {
    console.error('[SW] Sync failed:', error);
  }
}

// Push notifications (for future features)
self.addEventListener('push', event => {
  if (event.data) {
    const data = event.data.json();
    
    const options = {
      body: data.body || 'New update available',
      icon: '/placeholder.svg',
      badge: '/placeholder.svg',
      tag: data.tag || 'fishnet-notification',
      data: data.data || {},
      actions: [
        {
          action: 'open',
          title: 'Open App'
        },
        {
          action: 'dismiss',
          title: 'Dismiss'
        }
      ]
    };
    
    event.waitUntil(
      self.registration.showNotification(data.title || 'Fish Net', options)
    );
  }
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Message handling for communication with main app
self.addEventListener('message', event => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'GET_VERSION':
      event.ports[0].postMessage({
        version: CACHE_NAME,
        timestamp: Date.now()
      });
      break;
      
    case 'CLEAR_CACHE':
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => caches.delete(cacheName))
        );
      }).then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    default:
      console.log('[SW] Unknown message type:', type);
  }
});

console.log('[SW] Fish Net Service Worker loaded successfully');