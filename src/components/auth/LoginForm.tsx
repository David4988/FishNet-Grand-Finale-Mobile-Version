import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion'; // <-- DESIGN UPGRADE
import { Eye, EyeOff, Mail, Lock, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { authService } from '@/services/auth';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

// --- ANIMATION VARIANTS ---
const cardVariants = {
  hidden: { opacity: 0, y: 40, scale: 0.98 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.5, ease: 'easeOut' } },
};

const nameFieldVariants = {
  hidden: { opacity: 0, height: 0, y: -20, transition: { duration: 0.3, ease: 'easeOut' } },
  visible: { opacity: 1, height: 'auto', y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
};


export const LoginForm = () => {
  const { t } = useTranslation();
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: ''
  });
  
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      let result;
      if (isLogin) {
        result = await authService.signIn(formData.email, formData.password);
      } else {
        result = await authService.signUp(formData.email, formData.password, formData.name);
      }

      if (result.success) {
        toast({
          title: isLogin ? t('auth.welcomeBack') : t('common.success'),
          description: isLogin ? t('auth.signInMessage') : t('auth.signUpMessage'),
        });
      } else {
        toast({
          title: t('common.error'),
          description: result.error,
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('common.error'),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (field: keyof typeof formData) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }));
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 p-4 overflow-hidden">
      <div className="absolute inset-0 bg-grid-sky-400/[0.05]" />
      
      <motion.div initial="hidden" animate="visible" variants={cardVariants} className="w-full max-w-md">
        <Card className="w-full bg-gray-900/60 backdrop-blur-xl border border-sky-400/20 shadow-2xl shadow-sky-400/10">
          <CardHeader className="text-center space-y-4">
            <h1 className="text-4xl font-bold text-sky-400 tracking-wider">
              Fish Net
            </h1>
            <div>
              <CardTitle className="text-2xl font-bold text-white">
                {isLogin ? t('auth.welcomeBack') : t('auth.signUp')}
              </CardTitle>
              <CardDescription className="text-gray-400 mt-2">
                {isLogin 
                  ? t('auth.signInMessage') 
                  : t('auth.signUpMessage')}
              </CardDescription>
            </div>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <AnimatePresence>
                {!isLogin && (
                  <motion.div
                    key="name-field"
                    variants={nameFieldVariants}
                    initial="hidden"
                    animate="visible"
                    exit="hidden"
                    className="space-y-2 overflow-hidden"
                  >
                    <Label htmlFor="name" className="text-gray-300">{t('auth.name')}</Label>
                    <div className="relative group">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500 transition-colors group-focus-within:text-sky-400" />
                      <Input id="name" type="text" placeholder={t('auth.namePlaceholder')} value={formData.name} onChange={handleChange('name')} className="pl-10 bg-gray-800/50 border-gray-700 focus:border-sky-400 focus:ring-2 focus:ring-sky-400/20 text-white transition-all" required={!isLogin} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="space-y-2">
                <Label htmlFor="email" className="text-gray-300">{t('auth.email')}</Label>
                <div className="relative group">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500 transition-colors group-focus-within:text-sky-400" />
                  <Input id="email" type="email" placeholder={t('auth.emailPlaceholder')} value={formData.email} onChange={handleChange('email')} className="pl-10 bg-gray-800/50 border-gray-700 focus:border-sky-400 focus:ring-2 focus:ring-sky-400/20 text-white transition-all" required />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-gray-300">{t('auth.password')}</Label>
                <div className="relative group">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500 transition-colors group-focus-within:text-sky-400" />
                  <Input id="password" type={showPassword ? 'text' : 'password'} placeholder={t('auth.passwordPlaceholder')} value={formData.password} onChange={handleChange('password')} className="pl-10 pr-10 bg-gray-800/50 border-gray-700 focus:border-sky-400 focus:ring-2 focus:ring-sky-400/20 text-white transition-all" required />
                  <Button type="button" variant="ghost" size="sm" className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0" onClick={() => setShowPassword(!showPassword)}>
                    {showPassword ? <EyeOff className="h-4 w-4 text-gray-500" /> : <Eye className="h-4 w-4 text-gray-500" />}
                  </Button>
                </div>
              </div>

              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Button type="submit" disabled={isLoading} className={cn("w-full bg-sky-500 hover:bg-sky-600", "text-white font-semibold py-6 shadow-lg shadow-sky-500/20 transition-all duration-300", isLoading && "opacity-50 cursor-not-allowed")}>
                  {isLoading ? (
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                      {t('auth.loading')}
                    </div>
                  ) : (isLogin ? t('auth.signIn') : t('auth.signUp'))}
                </Button>
              </motion.div>

              <div className="text-center pt-4 border-t border-sky-400/10">
                <p className="text-gray-400">{isLogin ? t('auth.noAccount') : t('auth.haveAccount')}</p>
                <Button type="button" variant="link" onClick={() => setIsLogin(!isLogin)} className="text-sky-400 hover:text-sky-300 font-semibold p-0 h-auto mt-1">
                  {isLogin ? t('auth.signUpHere') : t('auth.signInHere')}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};