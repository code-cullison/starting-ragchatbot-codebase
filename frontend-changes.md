# Frontend Changes: Light/Dark Theme Toggle

## Summary
Added a circular icon button (sun/moon) fixed to the top-right that switches between the existing dark theme and a new light theme.

## Files changed
- `frontend/index.html`
  - Added `#themeToggle` button (inline SVG sun + moon icons, `aria-label`, `aria-pressed`) at the top of `<body>`.
  - Added a tiny inline script in `<head>` that applies the saved theme before first paint (no flash).
  - Bumped cache-busting query strings for `style.css` and `script.js`.
- `frontend/style.css`
  - Added `:root[data-theme="light"]` overriding the CSS variables (background, surface, text, border, links, etc.); dark remains the default.
  - New `--code-bg` variable replaces two hardcoded `rgba(0,0,0,0.2)` code-block backgrounds so they work in both themes.
  - `.theme-toggle` styles: fixed top-right, 44px round, matches surface/border/shadow tokens, hover and `:focus-visible` ring.
  - Icon animation: sun/moon cross-fade with rotate + scale (0.45s). Page colors animate via a temporary `html.theme-transition` class (0.35s), so normal interactions are not slowed.
  - `prefers-reduced-motion` disables the transitions.
- `frontend/script.js`
  - `setupThemeToggle()`: toggles `data-theme` on `<html>`, persists choice in `localStorage` (guarded with try/catch), and updates `aria-label`/`aria-pressed`.

## Accessibility
- Native `<button>`: reachable with Tab, activated with Enter/Space.
- Descriptive, state-aware `aria-label` ("Switch to light/dark theme"); icons are `aria-hidden`.
- Visible focus ring via `:focus-visible`; respects reduced-motion preference.
