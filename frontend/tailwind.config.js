/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        brand: {
          50:  '#EAFFF8',
          100: '#B4DED0',
          200: '#A3D5C4',
          300: '#6DCEAE',
          400: '#4E8C76',
          500: '#3F7867',
          600: '#16423C',
          700: '#0F302E',
        },
        surface: {
          page:  '#F0F0F0',
          card:  '#F3F3F3',
          white: '#FFFFFF',
          soft:  '#F1F6F4',
        },
        accent: {
          purple: '#8A38F5',
          orange: '#F59D38',
          gold:   '#FFC634',
          brightGold: '#F6DA07',
          red:    '#A10003',
          olive:  '#728D53',
        },
        neutral: {
          muted:    '#999999',
          border:   '#D9D9D9',
          black:    '#000000',
        },
      },
      boxShadow: {
        'glass':       '0 8px 32px rgba(22, 66, 60, 0.08)',
        'glass-lg':    '0 16px 48px rgba(22, 66, 60, 0.12)',
        'glass-inner': 'inset 0 1px 1px rgba(255, 255, 255, 0.4)',
        'card':        '0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06)',
        'card-hover':  '0 4px 16px rgba(22, 66, 60, 0.12), 0 8px 32px rgba(22, 66, 60, 0.08)',
        'float':       '0 20px 60px rgba(22, 66, 60, 0.15)',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      animation: {
        'fade-in':      'fadeIn 0.5s ease-out forwards',
        'slide-up':     'slideUp 0.5s ease-out forwards',
        'slide-down':   'slideDown 0.3s ease-out forwards',
        'scale-in':     'scaleIn 0.4s ease-out forwards',
        'float':        'float 6s ease-in-out infinite',
        'float-slow':   'floatSlow 8s ease-in-out infinite',
        'shimmer':      'shimmer 2s linear infinite',
        'pulse-soft':   'pulseSoft 3s ease-in-out infinite',
        'blob':         'blob 7s infinite',
        'count-up':     'countUp 1s ease-out forwards',
      },
      keyframes: {
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%':   { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%':   { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%':   { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%':      { transform: 'translateY(-20px)' },
        },
        floatSlow: {
          '0%, 100%': { transform: 'translateY(0) rotate(0deg)' },
          '50%':      { transform: 'translateY(-30px) rotate(5deg)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '0.6' },
          '50%':      { opacity: '1' },
        },
        blob: {
          '0%':   { transform: 'translate(0px, 0px) scale(1)' },
          '33%':  { transform: 'translate(30px, -50px) scale(1.1)' },
          '66%':  { transform: 'translate(-20px, 20px) scale(0.9)' },
          '100%': { transform: 'translate(0px, 0px) scale(1)' },
        },
        countUp: {
          '0%':   { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      backgroundImage: {
        'hero-gradient': 'linear-gradient(135deg, #F6F3C2 0%, #D1F0E4 50%, #EAFFF8 100%)',
        'nav-gradient':  'linear-gradient(135deg, #4E8C76, #0F302E)',
        'wave-gradient': 'linear-gradient(135deg, #A3D5C4, #16423C)',
        'footer-gradient': 'linear-gradient(135deg, #4E8C76, #0F302E)',
        'mint-gradient': 'linear-gradient(135deg, #B4DED0, #EAFFF8)',
        'teal-gradient': 'linear-gradient(135deg, #16423C, #3F7867)',
      },
    },
  },
  plugins: [],
}
