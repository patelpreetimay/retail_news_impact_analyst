import { useState } from 'react';
import {
  ChevronDown, TrendingUp, TrendingDown, Minus,
  Newspaper, ExternalLink, Clock, ChevronRight,
} from 'lucide-react';

const STANCE_CONFIG = {
  positive: {
    label: 'Bullish',
    icon: TrendingUp,
    bg: 'bg-emerald-50',
    text: 'text-emerald-700',
    ring: 'ring-emerald-200',
    dot: 'bg-emerald-500',
  },
  negative: {
    label: 'Bearish',
    icon: TrendingDown,
    bg: 'bg-red-50',
    text: 'text-red-700',
    ring: 'ring-red-200',
    dot: 'bg-red-500',
  },
  neutral: {
    label: 'Neutral',
    icon: Minus,
    bg: 'bg-gray-50',
    text: 'text-gray-600',
    ring: 'ring-gray-200',
    dot: 'bg-gray-400',
  },
};

const getImpactColor = (score) => {
  if (score >= 0.7) return 'text-brand-600';
  if (score >= 0.4) return 'text-accent-orange';
  return 'text-neutral-muted';
};

const getImpactBarWidth = (score) => `${Math.max(score * 100, 5)}%`;

const NewsCard = ({ item, isExpanded, onToggle, index }) => {
  const stance = STANCE_CONFIG[item.stance] || STANCE_CONFIG.neutral;
  const StanceIcon = stance.icon;

  return (
    <div
      className={`glass-solid rounded-2xl overflow-hidden transition-all duration-300 hover:shadow-card-hover group animate-slide-up opacity-0 ${
        isExpanded ? 'ring-1 ring-brand-300' : ''
      }`}
      style={{ animationDelay: `${Math.min(index * 60, 400)}ms`, animationFillMode: 'forwards' }}
    >
      {/* Main row */}
      <div
        className="p-4 sm:p-5 cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex items-start gap-4">
          {/* Impact indicator */}
          <div className="hidden sm:flex flex-col items-center gap-1 shrink-0 pt-0.5">
            <div className={`text-lg font-display font-bold ${getImpactColor(item.impact_score)}`}>
              {Number(item.impact_score).toFixed(2)}
            </div>
            <div className="w-8 h-1 rounded-full bg-gray-100 overflow-hidden">
              <div
                className="h-full rounded-full impact-bar transition-all duration-500"
                style={{ width: getImpactBarWidth(item.impact_score) }}
              />
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <h3 className="text-sm sm:text-base font-semibold text-neutral-black leading-snug mb-2 group-hover:text-brand-600 transition-colors line-clamp-2">
              {item.headline}
            </h3>

            <div className="flex flex-wrap items-center gap-2">
              {/* Event badge */}
              <span className="badge badge-event capitalize">
                {item.event_type?.replace(/_/g, ' ')}
              </span>

              {/* Stance badge */}
              <span className={`badge ${stance.bg} ${stance.text} ring-1 ${stance.ring}`}>
                <StanceIcon className="w-3 h-3 mr-1" />
                {stance.label}
              </span>

              {/* Source & time */}
              {item.source && (
                <span className="text-xs text-neutral-muted flex items-center gap-1">
                  <ExternalLink className="w-3 h-3" />
                  {item.source}
                </span>
              )}
              {item.timestamp && (
                <span className="text-xs text-neutral-muted flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {item.timestamp}
                </span>
              )}

              {/* Mobile impact */}
              <span className={`sm:hidden text-xs font-bold ${getImpactColor(item.impact_score)}`}>
                Impact: {Number(item.impact_score).toFixed(2)}
              </span>
            </div>
          </div>

          {/* Expand button */}
          <button
            className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
              isExpanded
                ? 'bg-brand-600 text-white rotate-180'
                : 'bg-surface-soft text-neutral-muted hover:bg-brand-50 hover:text-brand-600'
            }`}
          >
            <ChevronDown className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Expanded explanation */}
      {isExpanded && (
        <div className="px-4 sm:px-5 pb-4 sm:pb-5 animate-slide-down">
          <div className="bg-surface-soft rounded-xl p-4 border border-brand-100/50">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse-soft" />
              <span className="text-xs font-semibold text-brand-600 uppercase tracking-wider">Analysis Explanation</span>
            </div>
            <p className="text-sm text-neutral-black/80 leading-relaxed">
              {item.explanation || 'No detailed analysis available for this article.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

const NewsFeed = ({ data }) => {
  const [expandedId, setExpandedId] = useState(null);
  const [visibleCount, setVisibleCount] = useState(6);

  if (!data || data.length === 0) {
    return (
      <div className="glass-solid rounded-2xl p-12 text-center">
        <div className="w-14 h-14 rounded-2xl bg-surface-soft flex items-center justify-center mx-auto mb-4">
          <Newspaper className="w-7 h-7 text-neutral-muted" />
        </div>
        <h3 className="text-lg font-display font-semibold text-neutral-black mb-1">No articles found</h3>
        <p className="text-sm text-neutral-muted">Try adjusting your filters or refresh the data.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Newspaper className="w-4.5 h-4.5 text-brand-500" />
          <h2 className="text-base font-display font-semibold text-neutral-black">News Feed</h2>
        </div>
        <span className="text-xs font-medium text-neutral-muted bg-surface-soft px-3 py-1 rounded-full">
          {data.length} article{data.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Cards */}
      <div className="space-y-3">
        {data.slice(0, visibleCount).map((item, index) => (
          <NewsCard
            key={index}
            item={item}
            index={index}
            isExpanded={expandedId === index}
            onToggle={() => setExpandedId(expandedId === index ? null : index)}
          />
        ))}
      </div>

      {/* Load more */}
      {visibleCount < data.length && (
        <div className="flex justify-center mt-6">
          <button
            onClick={() => setVisibleCount(prev => prev + 6)}
            className="inline-flex items-center gap-2 px-6 py-2.5 text-sm font-semibold
                       text-brand-600 bg-white rounded-xl ring-1 ring-brand-200
                       hover:bg-brand-50 hover:ring-brand-300 active:scale-[0.98]
                       transition-all duration-200 shadow-sm"
          >
            Show more
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};

export default NewsFeed;
