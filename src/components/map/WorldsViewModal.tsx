import { Dialog, DialogContent } from '@/components/ui/dialog';
import { CatchMap } from '@/components/map/CatchMap';
import { NewsModule } from '@/components/map/NewsModule';
import { WeatherModule } from '@/components/map/WeatherModule';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { MapPin, Newspaper, Cloud } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface WorldsViewModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const WorldsViewModal = ({ open, onOpenChange }: WorldsViewModalProps) => {
  const { t } = useTranslation();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl w-[90vw] h-[90vh] p-0 overflow-hidden flex flex-col">

        <Tabs defaultValue="map" className="flex-1 flex flex-col">

          <TabsList className="grid w-full grid-cols-3 rounded-none border-b border-border">
            <TabsTrigger value="map" className="rounded-none flex items-center gap-2">
              <MapPin className="h-4 w-4" /> {t('map.tabs.map')}
            </TabsTrigger>

            <TabsTrigger value="news" className="rounded-none flex items-center gap-2">
              <Newspaper className="h-4 w-4" /> {t('map.tabs.news')}
            </TabsTrigger>

            <TabsTrigger value="weather" className="rounded-none flex items-center gap-2">
              <Cloud className="h-4 w-4" /> {t('map.tabs.weather')}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="map" className="flex-1 p-0 mt-0">
            <CatchMap className="h-full w-full" />
          </TabsContent>

          <TabsContent value="news" className="flex-1 p-0 mt-0">
            <NewsModule />
          </TabsContent>

          <TabsContent value="weather" className="flex-1 p-0 mt-0">
            <WeatherModule />
          </TabsContent>

        </Tabs>
      </DialogContent>
    </Dialog>
  );
};
