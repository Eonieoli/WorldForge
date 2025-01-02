/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        custom1: '#fff0dc',
        custom2: '#f0bb78',
        custom3: '#543a14',
      },
    },
  },
  plugins: [],
};
