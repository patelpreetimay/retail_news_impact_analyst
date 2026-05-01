import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts';

const CHART_COLORS = ['#16423C', '#3F7867', '#6DCEAE', '#4E8C76', '#A3D5C4', '#0F302E', '#728D53'];

const STANCE_COLORS = {
  positive: '#6DCEAE',
  negative: '#A10003',
  neutral:  '#999999',
};

const STANCE_LABELS = {
  positive: 'Bullish',
  negative: 'Bearish',
  neutral: 'Neutral',
};

// Custom tooltip
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-solid rounded-xl px-4 py-3 shadow-glass-lg text-sm">
      <p className="font-semibold text-neutral-black mb-0.5">
        {label || payload[0].name}
      </p>
      <p className="text-brand-600 font-bold text-lg">{payload[0].value}</p>
    </div>
  );
};

export const EventDistributionChart = ({ data }) => {
  if (!data || Object.keys(data).length === 0) return null;

  const chartData = Object.entries(data)
    .map(([name, value]) => ({
      name: name.replace(/_/g, ' '),
      value,
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="glass-solid rounded-2xl p-6 animate-slide-up opacity-0" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
      <div className="flex items-center gap-2 mb-6">
        <div className="w-3 h-3 rounded-full bg-brand-600" />
        <h3 className="text-sm font-display font-semibold text-neutral-black">Event Distribution</h3>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart layout="vertical" data={chartData} margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={true}
              vertical={false}
              stroke="#D9D9D9"
              strokeOpacity={0.5}
            />
            <XAxis
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#999999', fontSize: 11, fontWeight: 500 }}
            />
            <YAxis
              dataKey="name"
              type="category"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#000000', fontSize: 12, fontWeight: 500 }}
              width={130}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(22, 66, 60, 0.04)' }} />
            <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={20}>
              {chartData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={CHART_COLORS[index % CHART_COLORS.length]}
                  className="transition-opacity hover:opacity-80"
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const StanceDistributionChart = ({ data }) => {
  if (!data || Object.keys(data).length === 0) return null;

  const chartData = Object.entries(data).map(([name, value]) => ({
    name: STANCE_LABELS[name] || name,
    rawName: name,
    value,
  }));

  const total = chartData.reduce((sum, d) => sum + d.value, 0);

  return (
    <div className="glass-solid rounded-2xl p-6 animate-slide-up opacity-0" style={{ animationDelay: '300ms', animationFillMode: 'forwards' }}>
      <div className="flex items-center gap-2 mb-6">
        <div className="w-3 h-3 rounded-full bg-accent-purple" />
        <h3 className="text-sm font-display font-semibold text-neutral-black">Market Stance</h3>
      </div>

      <div className="h-64 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="45%"
              innerRadius={65}
              outerRadius={90}
              paddingAngle={3}
              dataKey="value"
              stroke="transparent"
              strokeWidth={0}
            >
              {chartData.map((entry) => (
                <Cell
                  key={entry.rawName}
                  fill={STANCE_COLORS[entry.rawName] || '#999999'}
                  className="transition-opacity hover:opacity-80"
                />
              ))}
            </Pie>
            <Tooltip content={<ChartTooltip />} />
            <Legend
              verticalAlign="bottom"
              height={36}
              iconType="circle"
              iconSize={8}
              formatter={(value) => (
                <span className="text-xs font-medium text-neutral-black ml-1">{value}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>

        {/* Center label */}
        <div className="absolute top-[38%] left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
          <p className="text-2xl font-display font-bold text-brand-600">{total}</p>
          <p className="text-[10px] text-neutral-muted font-medium uppercase tracking-wider">Total</p>
        </div>
      </div>
    </div>
  );
};
