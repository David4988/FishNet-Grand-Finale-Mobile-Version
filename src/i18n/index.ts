import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import en from './locales/en.json';
import ta from './locales/ta.json';
import te from './locales/te.json';
import hi from './locales/hi.json';
import kn from './locales/kn.json';
import ml from './locales/ml.json';
import gu from './locales/gu.json';
import mwr from './locales/mwr.json';
import bn from './locales/bn.json';
import pa from './locales/pa.json';
import mr from './locales/mr.json';
import or from './locales/or.json';

const resources = {
  en: { translation: en },
  ta: { translation: ta },
  te: { translation: te },
  hi: { translation: hi },
  kn: { translation: kn },
  ml: { translation: ml },
  gu: { translation: gu },
  mwr: { translation: mwr },
  bn: { translation: bn },
  pa: { translation: pa },
  mr: { translation: mr },
  or: { translation: or },
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    supportedLngs: ['en', 'ta', 'te', 'hi', 'kn', 'ml', 'gu', 'mwr', 'bn', 'pa', 'mr', 'or'],
    load: 'languageOnly',
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
      lookupLocalStorage: 'fishnet_language',
    },
  });

export default i18n;
