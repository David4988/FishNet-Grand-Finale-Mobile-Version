import { ReactNode, useEffect, useState } from 'react';
import { AuthState, authService } from '@/services/auth';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { LoginForm } from './LoginForm';

interface AuthGuardProps {
  children: ReactNode;
}

export const AuthGuard = ({ children }: AuthGuardProps) => {
  const [authState, setAuthState] = useState<AuthState>(authService.getState());

  useEffect(() => {
    const unsubscribe = authService.subscribe(setAuthState);
    return unsubscribe;
  }, []);

  if (authState.loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!authState.isAuthenticated) {
    return <LoginForm />;
  }

  return <>{children}</>;
};