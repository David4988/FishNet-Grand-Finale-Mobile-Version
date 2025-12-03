import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { MapPin, Globe, User, Mail, Loader2, Check, X } from 'lucide-react';
import { authService } from '@/services/auth';
import { preferencesService } from '@/services/preferences';
import { toast } from 'sonner';

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { t, i18n } = useTranslation();
  const currentUser = authService.getState().user;
  
  const [name, setName] = useState(currentUser?.name || '');
  const [email, setEmail] = useState(currentUser?.email || '');
  const [locationEnabled, setLocationEnabled] = useState(false);
  const [locationPermissionState, setLocationPermissionState] = useState<string>('unknown');
  const [currentLocation, setCurrentLocation] = useState<string>('');
  const [selectedLanguage, setSelectedLanguage] = useState(i18n.language);
  const [showLanguageConfirm, setShowLanguageConfirm] = useState(false);
  const [pendingLanguage, setPendingLanguage] = useState('');
  const [isRequestingLocation, setIsRequestingLocation] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const prefs = preferencesService.getPreferences();
    setLocationEnabled(prefs.locationEnabled);
    setLocationPermissionState(prefs.locationPermissionState);

    const unsubscribe = preferencesService.subscribe((prefs) => {
      setLocationEnabled(prefs.locationEnabled);
      setLocationPermissionState(prefs.locationPermissionState);
    });

    if (prefs.locationEnabled) {
      loadCurrentLocation();
    }

    return unsubscribe;
  }, []);

  const loadCurrentLocation = async () => {
    const result = await preferencesService.getCurrentLocation();
    if (result.success && result.position) {
      const { latitude, longitude } = result.position.coords;
      setCurrentLocation(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
    }
  };

  const handleLocationToggle = async () => {
    if (locationEnabled) {
      await preferencesService.disableLocation();
      setCurrentLocation('');
      toast.success(t('settings.locationDisabled'));
    } else {
      setIsRequestingLocation(true);
      const result = await preferencesService.requestLocationPermission();
      setIsRequestingLocation(false);

      if (result.success) {
        if (result.position) {
          const { latitude, longitude } = result.position.coords;
          setCurrentLocation(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
        }
        toast.success(t('settings.locationEnabled'));
      } else {
        toast.error(result.error || 'Failed to get location permission');
      }
    }
  };

  const handleLanguageChange = (value: string) => {
    setPendingLanguage(value);
    setShowLanguageConfirm(true);
  };

  const confirmLanguageChange = async () => {
    await preferencesService.setLanguage(pendingLanguage);
    await i18n.changeLanguage(pendingLanguage);
    setSelectedLanguage(pendingLanguage);
    setShowLanguageConfirm(false);
    toast.success(t('settings.languageChanged'));
    setTimeout(() => {
      window.location.reload();
    }, 1000);
  };

  const handleSave = async () => {
    if (!currentUser) return;

    setIsSaving(true);
    const result = await authService.updateProfile({
      name: name.trim() || currentUser.name,
      ...(email.trim() !== currentUser.email && { email: email.trim() })
    });

    setIsSaving(false);

    if (result.success) {
      toast.success(t('settings.saved'));
      onOpenChange(false);
    } else {
      toast.error(result.error || 'Failed to save settings');
    }
  };

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'ta', name: 'தமிழ்' },
    { code: 'te', name: 'తెలుగు' },
    { code: 'hi', name: 'हिन्दी' },
    { code: 'kn', name: 'ಕನ್ನಡ' },
    { code: 'ml', name: 'മലയാളം' },
    { code: 'gu', name: 'ગુજરાતી' },
    { code: 'mwr', name: 'मारवाड़ी' },
    { code: 'bn', name: 'বাংলা' },
    { code: 'pa', name: 'ਪੰਜਾਬੀ' },
    { code: 'mr', name: 'मराठी' },
    { code: 'or', name: 'ଓଡ଼ିଆ' },
  ];

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-[425px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t('settings.title')}</DialogTitle>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* Profile Settings */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <User className="h-4 w-4" />
                {t('settings.profile')}
              </h3>
              
              <div className="space-y-2">
                <Label htmlFor="name">{t('settings.name')}</Label>
                <Input
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={currentUser?.name}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">{t('settings.email')}</Label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder={currentUser?.email}
                />
              </div>
            </div>

            <Separator />

            {/* Location Settings */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <MapPin className="h-4 w-4" />
                {t('settings.location')}
              </h3>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>{t('settings.locationPermission')}</Label>
                  <p className="text-xs text-muted-foreground">
                    {locationEnabled ? t('settings.locationEnabled') : t('settings.locationDisabled')}
                  </p>
                </div>
                <Switch
                  checked={locationEnabled}
                  onCheckedChange={handleLocationToggle}
                  disabled={isRequestingLocation}
                />
              </div>

              {isRequestingLocation && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {t('settings.requestPermission')}
                </div>
              )}

              {currentLocation && (
                <div className="space-y-2">
                  <Label>{t('settings.currentLocation')}</Label>
                  <div className="flex items-center gap-2 text-sm bg-muted p-2 rounded-md">
                    <MapPin className="h-4 w-4 text-primary" />
                    <span className="font-mono">{currentLocation}</span>
                  </div>
                </div>
              )}

              {locationPermissionState === 'denied' && (
                <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-2 rounded-md">
                  <X className="h-4 w-4" />
                  <span>{t('settings.locationPermissionDenied', { defaultValue: 'Location permission denied. Please enable it in your browser settings.' })}</span>
                </div>
              )}
            </div>

            <Separator />

            {/* Language Settings */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Globe className="h-4 w-4" />
                {t('settings.language')}
              </h3>

              <div className="space-y-2">
                <Label htmlFor="language">{t('settings.selectLanguage')}</Label>
                <Select value={selectedLanguage} onValueChange={handleLanguageChange}>
                  <SelectTrigger id="language">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {languages.map((lang) => (
                      <SelectItem key={lang.code} value={lang.code}>
                        {lang.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <div className="flex gap-3 justify-end">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              {t('settings.cancel')}
            </Button>
            <Button onClick={handleSave} disabled={isSaving}>
              {isSaving ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  {t('common.loading')}
                </>
              ) : (
                <>
                  <Check className="h-4 w-4 mr-2" />
                  {t('settings.save')}
                </>
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Language Confirmation Dialog */}
      <AlertDialog open={showLanguageConfirm} onOpenChange={setShowLanguageConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('settings.confirmLanguage')}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('settings.confirmLanguageMessage', { 
                language: languages.find(l => l.code === pendingLanguage)?.name || pendingLanguage 
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('settings.cancel')}</AlertDialogCancel>
            <AlertDialogAction onClick={confirmLanguageChange}>
              {t('settings.confirm')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
