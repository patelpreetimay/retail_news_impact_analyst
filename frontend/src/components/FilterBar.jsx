import { SlidersHorizontal, X } from 'lucide-react';
import Dropdown from './Dropdown';

const FilterBar = ({
  eventFilter, setEventFilter,
  stanceFilter, setStanceFilter,
  sourceFilter, setSourceFilter,
  eventTypes, sources,
  onReset,
}) => {
  const hasActiveFilter =
    eventFilter !== 'all' || stanceFilter !== 'all' || sourceFilter !== 'all';

  const eventOptions = [
    { value: 'all', label: 'All Events' },
    ...eventTypes.map(et => ({ value: et, label: et.replace(/_/g, ' ') })),
  ];

  const stanceOptions = [
    { value: 'all',      label: 'All Stances' },
    { value: 'positive', label: 'Bullish' },
    { value: 'negative', label: 'Bearish' },
    { value: 'neutral',  label: 'Neutral' },
  ];

  const sourceOptions = [
    { value: 'all', label: 'All Sources' },
    ...sources.map(src => ({ value: src, label: src })),
  ];

  return (
    <div
      className="glass-solid rounded-2xl p-4 animate-slide-up opacity-0"
      style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}
    >
      <div className="flex flex-col lg:flex-row lg:items-center gap-3">
        {/* Label */}
        <div className="flex items-center gap-2 shrink-0">
          <SlidersHorizontal className="w-4 h-4 text-brand-500" />
          <span className="text-sm font-semibold text-neutral-black">Filters</span>
        </div>

        {/* Filter controls */}
        <div className="flex flex-wrap items-center gap-2 flex-1">
          <Dropdown
            value={eventFilter}
            onChange={setEventFilter}
            options={eventOptions}
            className="min-w-[160px]"
          />

          <Dropdown
            value={stanceFilter}
            onChange={setStanceFilter}
            options={stanceOptions}
            className="min-w-[140px]"
          />

          <Dropdown
            value={sourceFilter}
            onChange={setSourceFilter}
            options={sourceOptions}
            className="min-w-[160px]"
          />
        </div>

        {/* Reset */}
        {hasActiveFilter && (
          <button
            onClick={onReset}
            className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium
                       text-accent-red bg-red-50 rounded-lg hover:bg-red-100
                       transition-colors shrink-0"
          >
            <X className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>
    </div>
  );
};

export default FilterBar;
