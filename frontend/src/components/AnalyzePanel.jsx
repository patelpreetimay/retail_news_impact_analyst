import { useState } from 'react';
import { analyzeArticle } from '../api';
import {
  Loader2, Send, Sparkles, TrendingUp, TrendingDown, Minus,
  FileText, Gauge, Tag, MessageSquare,
} from 'lucide-react';

const STANCE_DISPLAY = {
  positive: { label: 'Bullish', icon: TrendingUp, color: 'text-emerald-600', bg: 'bg-emerald-50', ring: 'ring-emerald-200' },
  negative: { label: 'Bearish', icon: TrendingDown, color: 'text-red-600', bg: 'bg-red-50', ring: 'ring-red-200' },
  neutral:  { label: 'Neutral', icon: Minus, color: 'text-gray-600', bg: 'bg-gray-50', ring: 'ring-gray-200' },
};

const AnalyzePanel = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeArticle(text);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const stance = result ? (STANCE_DISPLAY[result.stance] || STANCE_DISPLAY.neutral) : null;

  return (
    <div className="glass-solid rounded-2xl overflow-hidden animate-slide-up opacity-0" style={{ animationDelay: '400ms', animationFillMode: 'forwards' }}>
      {/* Header with gradient accent */}
      <div className="relative px-6 pt-6 pb-4">
        <div className="absolute top-0 left-0 right-0 h-1 bg-wave-gradient rounded-t-2xl" />
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-xl bg-brand-50 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-brand-600" />
          </div>
          <div>
            <h2 className="text-base font-display font-semibold text-neutral-black">Analyze Article</h2>
            <p className="text-xs text-neutral-muted">Paste any financial news text for instant analysis</p>
          </div>
        </div>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="px-6 pb-6">
        <div className="relative mb-4">
          <textarea
            className="w-full p-4 text-sm text-neutral-black placeholder-neutral-muted/60
                       bg-surface-soft border border-neutral-border rounded-xl
                       focus:ring-2 focus:ring-brand-300 focus:border-brand-400
                       outline-none resize-none transition-all leading-relaxed"
            rows={4}
            placeholder="Enter a financial news article or headline to analyze..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <span className="absolute bottom-3 right-3 text-[10px] text-neutral-muted">
            {text.length} chars
          </span>
        </div>

        <div className="flex items-center justify-between">
          <p className="text-[11px] text-neutral-muted">
            Min 5 words required for accurate analysis
          </p>
          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="btn-primary inline-flex items-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                Analyze
              </>
            )}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="mx-6 mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
          <p className="text-sm text-accent-red font-medium">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mx-6 mb-6 animate-scale-in">
          <div className="bg-surface-soft border border-brand-100 rounded-2xl p-5">
            {/* Result header */}
            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-brand-100">
              <div className="w-2 h-2 rounded-full bg-brand-400 animate-pulse-soft" />
              <span className="text-xs font-semibold text-brand-600 uppercase tracking-wider">Analysis Result</span>
            </div>

            {/* Result cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
              {/* Event type */}
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex items-center gap-1.5 mb-2">
                  <Tag className="w-3.5 h-3.5 text-brand-500" />
                  <span className="text-[10px] font-semibold text-neutral-muted uppercase tracking-wider">Event Type</span>
                </div>
                <p className="text-sm font-bold text-neutral-black capitalize">
                  {result.event_type?.replace(/_/g, ' ')}
                </p>
              </div>

              {/* Stance */}
              <div className="bg-white rounded-xl p-4 shadow-sm">
                <div className="flex items-center gap-1.5 mb-2">
                  <FileText className="w-3.5 h-3.5 text-brand-500" />
                  <span className="text-[10px] font-semibold text-neutral-muted uppercase tracking-wider">Stance</span>
                </div>
                {stance && (
                  <div className="flex items-center gap-1.5">
                    <stance.icon className={`w-4 h-4 ${stance.color}`} />
                    <span className={`text-sm font-bold ${stance.color}`}>{stance.label}</span>
                  </div>
                )}
              </div>

              {/* Impact score */}
              <div className="bg-white rounded-xl p-4 shadow-sm relative overflow-hidden">
                <div className="flex items-center gap-1.5 mb-2">
                  <Gauge className="w-3.5 h-3.5 text-brand-500" />
                  <span className="text-[10px] font-semibold text-neutral-muted uppercase tracking-wider">Impact</span>
                </div>
                <p className="text-2xl font-display font-bold text-brand-600">
                  {result.impact_score?.toFixed(2)}
                </p>
                {/* Mini bar */}
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-100">
                  <div
                    className="h-full impact-bar transition-all duration-700"
                    style={{ width: `${result.impact_score * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Explanation */}
            <div className="bg-white rounded-xl p-4 shadow-sm">
              <div className="flex items-center gap-1.5 mb-2">
                <MessageSquare className="w-3.5 h-3.5 text-brand-500" />
                <span className="text-[10px] font-semibold text-neutral-muted uppercase tracking-wider">Explanation</span>
              </div>
              <p className="text-sm text-neutral-black/80 leading-relaxed">
                {result.explanation}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyzePanel;
