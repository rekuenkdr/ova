/**
 * Theme Module - Handle light/dark theme switching
 */

const THEME_KEY = 'ova-theme';
const THEMES = ['dark', 'light', 'her', 'hal-9000'];
const DEFAULT_THEME = 'dark';

let currentTheme = DEFAULT_THEME;

/**
 * Get the saved theme from localStorage or detect system preference
 */
function getSavedTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved && THEMES.includes(saved)) {
    return saved;
  }
  // Detect system preference
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    return 'light';
  }
  return DEFAULT_THEME;
}

/**
 * Apply theme to the document
 */
function applyTheme(theme) {
  if (!THEMES.includes(theme)) {
    theme = DEFAULT_THEME;
  }

  currentTheme = theme;

  // Set data-theme attribute on html element
  document.documentElement.setAttribute('data-theme', theme);

  // Update theme CSS file
  const themeLink = document.getElementById('theme-css');
  if (themeLink) {
    themeLink.href = `static/themes/${theme}/theme.css`;
  }

  // Save to localStorage
  localStorage.setItem(THEME_KEY, theme);

  // Update theme toggle buttons if they exist
  updateThemeButtons();
}

/**
 * Toggle to next theme in cycle
 */
function toggleTheme() {
  const currentIndex = THEMES.indexOf(currentTheme);
  const nextIndex = (currentIndex + 1) % THEMES.length;
  applyTheme(THEMES[nextIndex]);
}

/**
 * Update theme toggle button states in settings panel
 */
function updateThemeButtons() {
  THEMES.forEach(theme => {
    const btn = document.querySelector(`.theme-toggle-btn[data-theme="${theme}"]`);
    if (btn) {
      btn.classList.toggle('active', currentTheme === theme);
    }
  });
}

/**
 * Get current theme
 */
function getTheme() {
  return currentTheme;
}

/**
 * Initialize theme system
 */
function initTheme() {
  // Apply saved/detected theme
  const theme = getSavedTheme();
  applyTheme(theme);

  // Listen for system theme changes
  if (window.matchMedia) {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', (e) => {
      // Only auto-switch if user hasn't manually set a preference
      const saved = localStorage.getItem(THEME_KEY);
      if (!saved) {
        applyTheme(e.matches ? 'dark' : 'light');
      }
    });
  }
}

// Export for use in other modules
export { initTheme, applyTheme, toggleTheme, getTheme, THEMES };
