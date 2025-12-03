// src/utils/localCatches.ts
export type LocalCatch = {
  id: string;
  createdAt: number;
  species: string;
  speciesLabel?: string;
  image: string;   // base64 or blob URL
  lat: number;
  lng: number;
  healthScore?: number;
  confidence?: number;
  estimatedWeight?: number;
};

const KEY = 'localCatches';

export function loadLocalCatches(): LocalCatch[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '[]');
  } catch {
    return [];
  }
}

export function addLocalCatch(c: LocalCatch) {
  const list = loadLocalCatches();
  list.unshift(c); // newest first
  localStorage.setItem(KEY, JSON.stringify(list));
}

export function clearLocalCatches() {
  localStorage.removeItem(KEY);
}
