import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface MobileSafeAreaProps {
  children: ReactNode;
  className?: string;
  top?: boolean;
  bottom?: boolean;
}

export const MobileSafeArea = ({ 
  children, 
  className, 
  top = true, 
  bottom = true 
}: MobileSafeAreaProps) => {
  return (
    <div className={cn(
      "w-full",
      top && "pt-safe-top",
      bottom && "pb-safe-bottom",
      className
    )}>
      {children}
    </div>
  );
};

// Mobile-first container with proper spacing
export const MobileContainer = ({ 
  children, 
  className 
}: { 
  children: ReactNode; 
  className?: string; 
}) => {
  return (
    <div className={cn(
      "container mx-auto px-4 sm:px-6 max-w-md",
      className
    )}>
      {children}
    </div>
  );
};

// Touch-friendly button wrapper
export const TouchArea = ({ 
  children, 
  className,
  onClick
}: { 
  children: ReactNode; 
  className?: string;
  onClick?: () => void;
}) => {
  return (
    <div 
      className={cn(
        "min-h-[44px] min-w-[44px] flex items-center justify-center touch-manipulation",
        onClick && "cursor-pointer active:scale-95 transition-transform",
        className
      )}
      onClick={onClick}
    >
      {children}
    </div>
  );
};