// Convert an array of normalized articles to CSV and trigger a browser download.
export const exportArticlesCSV = (articles, filename = 'rnia-articles.csv') => {
  if (!articles || articles.length === 0) {
    alert('No articles to export.');
    return;
  }

  const columns = [
    { key: 'id',            label: 'ID' },
    { key: 'headline',      label: 'Headline' },
    { key: 'event_type',    label: 'Event Type' },
    { key: 'event_label',   label: 'Event Label' },
    { key: 'stance',        label: 'Stance' },
    { key: 'impact_score',  label: 'Impact Score' },
    { key: 'materiality',   label: 'Materiality' },
    { key: 'credibility',   label: 'Credibility' },
    { key: 'recency',       label: 'Recency' },
    { key: 'source_name',   label: 'Source' },
    { key: 'timestamp',     label: 'Published' },
    { key: 'url',           label: 'URL' },
    { key: 'explanation',   label: 'Explanation' },
  ];

  const escape = (v) => {
    if (v === null || v === undefined) return '';
    const s = String(v).replace(/"/g, '""');
    return /[",\n\r]/.test(s) ? `"${s}"` : s;
  };

  const header = columns.map(c => c.label).join(',');
  const rows = articles.map(a =>
    columns.map(c => escape(a[c.key])).join(',')
  );
  const csv = '﻿' + [header, ...rows].join('\r\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
