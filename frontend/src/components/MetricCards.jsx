import { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Target, Activity } from 'lucide-react';

// Animated counter hook
function useCountUp(end, duration = 1200) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (typeof end !== 'number' || isNaN(end)) {
      setCount(0);
      return;
    }

    let startTime = null;

    function step(timestamp) {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setCount(end * eased);
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  }, [end, duration]);

  return count;
}

const MetricCardShell = ({ icon: Icon, label, color, delay, children }) => (
  <div
    className="glass-solid rounded-2xl p-5 group hover:shadow-card-hover hover:-translate-y-0.5 transition-all duration-300 animate-slide-up opacity-0"
    style={{ animationDelay: `${delay}ms`, animationFillMode: 'forwards' }}
  >
    <div className="flex items-start justify-between mb-4">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div className="w-8 h-1 rounded-full bg-neutral-border/50 group-hover:bg-brand-300 transition-colors" />
    </div>
    <p className="text-xs font-medium text-neutral-muted uppercase tracking-wider mb-1.5">{label}</p>
    {children}
  </div>
);

const NumberValue = ({ value, float = false, suffix }) => {
  const animated = useCountUp(typeof value === 'number' ? value : 0);
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-3xl font-display font-bold text-neutral-black">
        {float ? animated.toFixed(2) : Math.round(animated)}
      </span>
      {suffix && <span className="text-sm text-neutral-muted font-medium">{suffix}</span>}
    </div>
  );
};

const MetricCards = ({ summary }) => {
  if (!summary) return null;

  const stanceEntries = Object.entries(summary.stance_distribution || {});
  const bullishCount = stanceEntries.find(([k]) => k === 'positive')?.[1] || 0;
  const bearishCount = stanceEntries.find(([k]) => k === 'negative')?.[1] || 0;
  const neutralCount = stanceEntries.find(([k]) => k === 'neutral')?.[1]  || 0;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCardShell
        icon={BarChart3}
        label="Total Articles"
        color="bg-brand-50 text-brand-600"
        delay={0}
      >
        <NumberValue value={summary.total_articles} />
      </MetricCardShell>

      <MetricCardShell
        icon={TrendingUp}
        label="Avg Impact Score"
        color="bg-orange-50 text-accent-orange"
        delay={100}
      >
        <NumberValue value={summary.average_impact_score} float suffix="/ 1.0" />
      </MetricCardShell>

      <MetricCardShell
        icon={Target}
        label="Top Catalyst"
        color="bg-purple-50 text-accent-purple"
        delay={200}
      >
        <p
          className="text-xl font-display font-bold text-neutral-black capitalize truncate"
          title={summary.top_event}
        >
          {summary.top_event?.replace(/_/g, ' ') || 'N/A'}
        </p>
      </MetricCardShell>

      <MetricCardShell
        icon={Activity}
        label="Market Stance"
        color="bg-emerald-50 text-emerald-600"
        delay={300}
      >
        <div className="flex items-center gap-3 mt-0.5">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="text-sm font-semibold text-emerald-600">{bullishCount}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            <span className="text-sm font-semibold text-red-600">{bearishCount}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-gray-400" />
            <span className="text-sm font-semibold text-gray-500">{neutralCount}</span>
          </div>
        </div>
      </MetricCardShell>
    </div>
  );
};

export default MetricCards;
