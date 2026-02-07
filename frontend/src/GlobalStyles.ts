import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  :root {
    /* Modern Fintech Palette */
    --color-bg: #ffffff;           /* Pure White */
    --color-surface: #f8fafc;      /* Very Light Gray/Blue */
    --color-primary: #2563eb;      /* Royal Blue (Trust) */
    --color-primary-dark: #1d4ed8; /* Darker Blue for hover */
    --color-text-main: #0f172a;    /* Navy Black */
    --color-text-secondary: #64748b; /* Slate Gray */
    --color-accent: #3b82f6;       
    
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    
    --font-main: 'Inter', system-ui, -apple-system, sans-serif;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    background-color: var(--color-bg);
    color: var(--color-text-main);
    font-family: var(--font-main);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  /* Smooth Scrolling */
  html {
    scroll-behavior: smooth;
  }
`;