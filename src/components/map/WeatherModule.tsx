import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2 } from 'lucide-react';

export const WeatherModule = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);

  return (
    <div className="relative w-full h-full">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background z-10">
          <div className="text-center">
            <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto mb-4" />
            <p className="text-muted-foreground">{t('map.weather.loading')}</p>
          </div>
        </div>
      )}
      <iframe
        src="https://final-final-weather.onrender.com/"
        className="w-full h-full border-0"
        title={t('map.weather.title')}
        onLoad={() => setLoading(false)}
      />
    </div>
  );
};
