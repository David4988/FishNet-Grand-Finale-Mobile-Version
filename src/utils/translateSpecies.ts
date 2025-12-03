import { TFunction } from 'i18next';

/**
 * Translates fish species names based on the current language
 * @param speciesName - The English species name
 * @param t - The translation function from i18next
 * @returns Translated species name or original if no translation exists
 */
export function translateSpecies(speciesName: string, t: TFunction): string {
  if (!speciesName) return speciesName;
  
  // Try to get translation from species namespace
  const translationKey = `species.${speciesName}`;
  const translated = t(translationKey);
  
  // If translation exists and is different from the key, return it
  // Otherwise return the original name
  return translated !== translationKey ? translated : speciesName;
}
