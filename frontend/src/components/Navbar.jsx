import { RefreshCcw, Loader2, TrendingUp, Clock } from 'lucide-react';

const Navbar = ({ loading, onRefresh, lastRefreshed }) => {
  return (
    <nav className="sticky top-0 z-50 glass-nav">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-teal-gradient flex items-center justify-center shadow-md">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <div className="flex flex-col">
              <span className="text-lg font-display font-bold text-brand-600 leading-tight tracking-tight">
                RNIA
              </span>
              <span className="text-[10px] text-neutral-muted font-medium leading-none hidden sm:block">
                Retail News Impact Analyst
              </span>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Last synced */}
            <div className="hidden md:flex items-center gap-1.5 text-xs text-neutral-muted bg-surface-soft px-3 py-1.5 rounded-lg">
              <Clock className="w-3.5 h-3.5" />
              <span>Synced {lastRefreshed.toLocaleTimeString()}</span>
            </div>

            {/* Refresh */}
            <button
              onClick={onRefresh}
              disabled={loading}
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold
                         text-white bg-brand-600 rounded-xl
                         hover:bg-brand-700 active:scale-[0.97]
                         transition-all duration-200
                         shadow-sm hover:shadow-md
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="hidden sm:inline">Syncing...</span>
                </>
              ) : (
                <>
                  <RefreshCcw className="w-4 h-4" />
                  <span className="hidden sm:inline">Refresh</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
