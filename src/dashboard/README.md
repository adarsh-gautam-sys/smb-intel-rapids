# SMB Intelligence Dashboard
Stitch MCP generated UI — 4 pages, all fixed and linked.

## Pages
| File | Route | Active Tab |
|------|-------|------------|
| index.html | /dashboard/index.html | Overview |
| skus.html | /dashboard/skus.html | SKU Intelligence |
| cashflow.html | /dashboard/cashflow.html | Cash Flow |
| benchmark.html | /dashboard/benchmark.html | Benchmark |

## Nav Links
All 4 tabs are wired in every page. Click any tab to navigate.

## API Integration
- `window.API_BASE` auto-detects localhost vs Cloud Run
- All fetch() calls use `API_BASE + '/api/...'`
- Graceful fallback to mock data if API unreachable

## Cloud Run
Served as static files from FastAPI at `/dashboard/`.
Root URL `/` redirects to `/dashboard/index.html`.
