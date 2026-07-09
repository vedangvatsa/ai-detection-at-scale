# AI Detector — Article Website

This folder contains a long-form article and a React page component for `https://veda.ng/ai-detector`.

## Files

- `content.md` — the article text in Markdown. Use this if you want to render it with a static-site generator.
- `AIDetectorPage.tsx` — a self-contained React component with inline visuals (feature grid, comparison chart, benchmark cards, detector table, audit list).
- `pages/index.tsx` — a tiny Next.js page wrapper.

## How to use it

### Option 1: Drop into an existing Next.js site

Copy `AIDetectorPage.tsx` into your site (for example, `app/ai-detector/page.tsx` or `pages/ai-detector.tsx`) and render it:

```tsx
import AIDetectorPage from './AIDetectorPage';

export default function Page() {
  return <AIDetectorPage />;
}
```

The component uses Tailwind CSS utility classes. If your site uses Tailwind, it will inherit your theme. If not, you can replace the classes with your own styling.

### Option 2: Use the Markdown content

Copy `content.md` into your CMS or static-site generator. It is written in plain Markdown with no custom syntax.

### Option 3: Convert the React component to plain HTML/CSS

If you do not use React, you can paste `content.md` into an HTML template and recreate the SVG chart by hand. The chart data is:

```text
11-feature AUC: 0.9645
35-feature AUC: 0.9826
11-feature Accuracy: 0.9011
35-feature Accuracy: 0.9361
```

## Notes

- The article is written in the first person from the project owner's perspective.
- All benchmark numbers come from `results/RESULTS_SUMMARY.md` in this repository.
- The tone is explanatory and honest about limitations.
