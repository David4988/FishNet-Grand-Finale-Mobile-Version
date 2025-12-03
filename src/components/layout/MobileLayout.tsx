import { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { BottomNavigation } from './BottomNavigation'; // Assuming BottomNavigation is in the same folder
// import { Header } from './Header'; // Assuming a Header component exists

interface MobileLayoutProps {
  children: ReactNode;
  className?: string;
  showBottomNav?: boolean;
  showHeader?: boolean; // New prop to control header visibility
}

export const MobileLayout = ({ children, className, showBottomNav = true, showHeader = true }: MobileLayoutProps) => {
  return (
    // Main wrapper fills the dynamic viewport and prevents any global overflow.
    <div className={cn(
      "relative h-[100dvh] w-full overflow-hidden bg-slate-900",
      className
    )}>
      
      {/* Conditionally render the Header */}
      {/* {showHeader && (
        <Header />
      )} */}
      
      {/* Main Content: The Scrollable Zone */}
      <main className={cn(
        "absolute left-0 right-0 overflow-y-auto",
        // --- FIX: Conditionally apply top padding ---
        // If the header is shown, push content down. If not, content starts at the top.
        showHeader ? "top-0" : "top-0",
        showBottomNav ? "bottom-20" : "bottom-0"
      )}>
        {children}
      </main>

      {/* Conditionally render the Bottom Navigation */}
      {showBottomNav && (
        <footer className="absolute bottom-0 left-0 right-0 z-40 h-20 bg-slate-900/60 backdrop-blur-xl border-t border-sky-400/20">
          <BottomNavigation />
        </footer>
      )}
    </div>
  );
};