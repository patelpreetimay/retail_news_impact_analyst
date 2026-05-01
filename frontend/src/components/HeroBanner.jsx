import { BarChart3, TrendingUp, Zap, Globe } from 'lucide-react';

const HeroBanner = ({ summary }) => {
  return (
    <section className="relative overflow-hidden rounded-3xl bg-hero-gradient p-6 sm:p-8 lg:p-10">
      {/* Animated blobs */}
      <div className="blob blob-teal w-48 h-48 -top-10 -right-10 animate-blob opacity-20" />
      <div className="blob blob-mint w-64 h-64 -bottom-20 -left-16 animate-blob opacity-15" style={{ animationDelay: '2s' }} />
      <div className="blob blob-gold w-32 h-32 top-1/2 right-1/4 animate-blob opacity-10" style={{ animationDelay: '4s' }} />

      <div className="relative z-10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-display font-bold text-brand-700 mb-1">
              Market Intelligence
            </h1>
            <p className="text-sm text-brand-500/80 font-medium">
              Real-time financial news analysis & impact scoring
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/60 backdrop-blur-sm text-xs font-semibold text-brand-600 ring-1 ring-brand-200/50">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-soft" />
              Live
            </span>
          </div>
        </div>

        {/* Quick stat pills */}
        {summary && (
          <div className="flex flex-wrap gap-3">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl bg-white/50 backdrop-blur-sm shadow-sm">
              <div className="w-8 h-8 rounded-lg bg-brand-600/10 flex items-center justify-center">
                <BarChart3 className="w-4 h-4 text-brand-600" />
              </div>
              <div>
                <p className="text-[10px] font-medium text-neutral-muted uppercase tracking-wider">Articles</p>
                <p className="text-lg font-bold text-brand-700 leading-tight">{summary.total_articles}</p>
              </div>
            </div>

            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl bg-white/50 backdrop-blur-sm shadow-sm">
              <div className="w-8 h-8 rounded-lg bg-accent-orange/10 flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-accent-orange" />
              </div>
              <div>
                <p className="text-[10px] font-medium text-neutral-muted uppercase tracking-wider">Avg Impact</p>
                <p className="text-lg font-bold text-brand-700 leading-tight">{summary.average_impact_score}</p>
              </div>
            </div>

            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl bg-white/50 backdrop-blur-sm shadow-sm">
              <div className="w-8 h-8 rounded-lg bg-accent-purple/10 flex items-center justify-center">
                <Zap className="w-4 h-4 text-accent-purple" />
              </div>
              <div>
                <p className="text-[10px] font-medium text-neutral-muted uppercase tracking-wider">Top Event</p>
                <p className="text-sm font-bold text-brand-700 leading-tight capitalize">
                  {summary.top_event?.replace(/_/g, ' ')}
                </p>
              </div>
            </div>

            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl bg-white/50 backdrop-blur-sm shadow-sm">
              <div className="w-8 h-8 rounded-lg bg-accent-olive/10 flex items-center justify-center">
                <Globe className="w-4 h-4 text-accent-olive" />
              </div>
              <div>
                <p className="text-[10px] font-medium text-neutral-muted uppercase tracking-wider">Sources</p>
                <p className="text-lg font-bold text-brand-700 leading-tight">
                  {Object.keys(summary.stance_distribution || {}).length}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default HeroBanner;
