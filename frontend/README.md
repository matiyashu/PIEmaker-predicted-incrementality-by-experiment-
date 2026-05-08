# PIEmaker — Frontend

Next.js 14 dashboard for the PIE Measurement Workbench. Built by Prima Hanura Akbar.

## Local development

```bash
npm install
npm run dev    # http://localhost:3765
```

The frontend talks to the FastAPI backend via Next.js rewrites (`/api/backend/*` → `NEXT_PUBLIC_BACKEND_URL`, default `http://localhost:8000`).

## Demo mode

The dashboard runs entirely on canned mock data when demo mode is active. Useful for:

- offline previews
- screenshots
- Vercel deployments where the FastAPI backend is offline

Three ways to activate it:

| Method | Where | Persistence |
|---|---|---|
| URL flag | `?demo=1` on any route | Sticks via `localStorage` (`pie:demo=1`) |
| Settings page | `/settings` toggle | Sticks via `localStorage` |
| Build-time env var | `NEXT_PUBLIC_FORCE_DEMO=1` | Always on for the deployed bundle |

When demo mode is active, every workbench page renders with realistic mock data — no upload IDs needed.

## Vercel deployment

```bash
# from the repo root, deploy frontend/ as the project directory
cd frontend
vercel
```

`vercel.json` is pre-configured to set `NEXT_PUBLIC_FORCE_DEMO=1`, so the deployed app runs in demo mode by default. To wire it to a live backend instead:

1. Remove the `NEXT_PUBLIC_FORCE_DEMO` lines from `vercel.json`.
2. Set `NEXT_PUBLIC_BACKEND_URL=https://your-backend.fly.dev` (or wherever you've hosted the FastAPI service) in Vercel project settings.
3. Re-deploy.

The backend itself can't run on Vercel's serverless platform (it's a long-running stateful FastAPI app with file-based persistence). Host it on Render, Fly, or your own VPS.

## Routes

```
/                  Landing page (with "Try demo mode" CTA)
/dashboard         Hero metrics, charts, seed CTA
/upload            CSV upload + schema mapping
/cleaning          Cleaning audit trail with MC defense
/donor-pool        Promote/demote RCTs, coverage, aging, shadow recs
/labels            ATT / IC / ICPD per RCT
/features          X_pre + X_post engineering
/models            Train, ablation, hold-out-one-level, promote
/predict           Single-campaign forecast
/portfolio         Bulk-score every non-RCT in an upload
/decisions         Risk-gated promote / hold / deprioritize / block
/drift             PSI per feature with retrain verdict
/simulator         What-if budget reallocation
/settings          Demo toggle, backend URL, localStorage
/faq               Methodology, glossary, common questions
```

## Tech stack

- Next.js 14 App Router (route groups: `(workbench)/` for sidebar layout, `/` for landing)
- Tailwind CSS + shadcn-style design tokens
- Recharts for charts
- lucide-react for icons
- Single-dispatcher API mock layer (`lib/api.ts:_mockDispatch`) — keeps types tight and dependency-free
