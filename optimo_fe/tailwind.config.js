/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      keyframes: {
        glow: {
          '0%, 100%': { opacity: '0.1' },
          '50%': { opacity: '1' },
        },
        slideup: {
          '0%': { opacity: '0', transform: 'translateY(40px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        glow: 'glow 3.5s ease-in-out infinite',
        slideup: 'slideup 0.8s cubic-bezier(0.4,0,0.2,1) both',
      },
    },
  },
  plugins: [],
};
