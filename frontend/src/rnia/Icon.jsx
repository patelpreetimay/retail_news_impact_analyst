import {
  LayoutGrid, AlignJustify, Sparkles, Compass, Radio, Bookmark, FileText,
  BookOpen, Settings, Search, Bell, RotateCw, Moon, Sun, ChevronRight,
  ArrowUp, ArrowDown, ArrowRight, Minus, TrendingUp, Activity, Layers,
  Plus, Play, Download, Copy, Check, Star, SlidersHorizontal, Filter,
  Eye,
} from 'lucide-react';

const MAP = {
  grid: LayoutGrid,
  feed: AlignJustify,
  sparkles: Sparkles,
  compass: Compass,
  radio: Radio,
  bookmark: Bookmark,
  file: FileText,
  book: BookOpen,
  settings: Settings,
  search: Search,
  bell: Bell,
  refresh: RotateCw,
  moon: Moon,
  sun: Sun,
  chevron: ChevronRight,
  arrowUp: ArrowUp,
  arrowDown: ArrowDown,
  arrowRight: ArrowRight,
  minus: Minus,
  trending: TrendingUp,
  pulse: Activity,
  layers: Layers,
  plus: Plus,
  play: Play,
  download: Download,
  copy: Copy,
  check: Check,
  star: Star,
  sliders: SlidersHorizontal,
  filter: Filter,
  eye: Eye,
};

export const Icon = ({ name, size = 16, ...rest }) => {
  const C = MAP[name];
  if (!C) return null;
  return <C size={size} strokeWidth={1.5} {...rest} />;
};
