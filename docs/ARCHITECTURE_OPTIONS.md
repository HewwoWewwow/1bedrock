# Production Architecture Options

This document outlines options for rebuilding the Austin TIF Model prototype into a production-grade application that can handle concurrency, API integrations, database operations, and web traffic.

## Current State

- **Framework:** Streamlit (Python)
- **Calculations:** Pure Python with numpy_financial
- **Data:** In-memory session state
- **Hosting:** Local development

## Requirements for Production

- Handle concurrent users (10-100+)
- Read/write to databases (save scenarios, user data)
- Pull data from external APIs
- Private access with authentication
- Reliable hosting with good uptime

---

## Option 1: FastAPI + React/Next.js (Recommended)

**Keep your Python calculations, add a modern frontend**

### Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  React/Next.js  │────▶│    FastAPI      │────▶│  PostgreSQL  │
│   Frontend      │     │  (Python calcs) │     │   Database   │
└─────────────────┘     └─────────────────┘     └──────────────┘
```

### Pros
- Reuse 90% of existing Python calculation code
- FastAPI handles concurrency natively (async)
- Modern, fast, great developer experience
- Auto-generated API documentation (Swagger/OpenAPI)
- Clear separation of concerns
- Scales well

### Cons
- Need to learn React (or hire someone)
- Two codebases to maintain (frontend + backend)
- More deployment complexity

### Effort Estimate
- 1 week: Wrap existing calculations in FastAPI endpoints
- 3-6 weeks: Build React/Next.js frontend
- 1 week: Database integration, authentication

### Tech Stack
- **Backend:** FastAPI, SQLAlchemy, Pydantic
- **Frontend:** Next.js or React + Vite
- **Database:** PostgreSQL
- **Auth:** Auth0, Clerk, or FastAPI-Users
- **Hosting:** Railway, Render, or AWS

---

## Option 2: Django + HTMX

**Full Python stack with minimal JavaScript**

### Architecture
```
┌─────────────────────────────────────────┐     ┌──────────────┐
│              Django                      │────▶│  PostgreSQL  │
│  (Templates + HTMX for interactivity)   │     │   Database   │
└─────────────────────────────────────────┘     └──────────────┘
```

### Pros
- All Python - no JavaScript framework to learn
- Built-in authentication, admin panel, ORM
- Battle-tested for business applications
- HTMX provides interactive UI without React complexity
- Single codebase

### Cons
- Heavier framework with steeper learning curve
- Less "modern" feel than React SPAs
- HTMX has limitations for complex interactivity

### Effort Estimate
- 4-6 weeks total
- Calculations can be imported directly

### Tech Stack
- **Framework:** Django 5.x
- **Interactivity:** HTMX + Alpine.js
- **Database:** PostgreSQL (built-in ORM)
- **Auth:** Django built-in
- **Hosting:** Railway, Render, Heroku

---

## Option 3: Next.js Full-Stack (TypeScript)

**Rewrite everything in JavaScript/TypeScript**

### Architecture
```
┌─────────────────────────────────────────┐     ┌──────────────┐
│            Next.js (Vercel)             │────▶│  PostgreSQL  │
│  (React frontend + API routes)          │     │  (via Prisma)│
└─────────────────────────────────────────┘     └──────────────┘
```

### Pros
- One language (TypeScript) everywhere
- Excellent hosting on Vercel (near-zero config)
- Great performance with server components
- Large ecosystem and community

### Cons
- Must rewrite ALL calculations in TypeScript
- Lose Python ecosystem (numpy, pandas, numpy_financial)
- Financial calculations in JS require careful handling

### Effort Estimate
- 6-10 weeks (calculation rewrite is the bottleneck)
- High risk of introducing calculation bugs

### Tech Stack
- **Framework:** Next.js 14+ (App Router)
- **Database:** PostgreSQL via Prisma
- **Auth:** NextAuth.js or Clerk
- **Hosting:** Vercel (optimized for Next.js)

---

## Option 4: Streamlit + FastAPI Hybrid

**Keep Streamlit UI, add API/database layer**

### Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────┐
│    Streamlit    │────▶│    FastAPI      │────▶│  PostgreSQL  │
│   (existing UI) │     │ (data layer)    │     │   Database   │
└─────────────────┘     └─────────────────┘     └──────────────┘
```

### Pros
- Minimal changes to current code
- Add database/API capabilities incrementally
- Fastest path to "production-ish"
- Good for internal tools with limited users

### Cons
- Streamlit has scalability limits (~50 concurrent users)
- Not ideal for public-facing applications
- Session state management can get complex

### Effort Estimate
- 1-2 weeks for basic database integration
- Additional time for API integrations as needed

### Tech Stack
- **UI:** Streamlit (existing)
- **Backend:** FastAPI for data operations
- **Database:** PostgreSQL or SQLite
- **Hosting:** Streamlit Cloud + separate API host

---

## Recommendation

### For Your Use Case (Financial Model, Private Sharing, ~10-50 Users)

**Short-term (Now):**
- Deploy current Streamlit app via ngrok for immediate feedback
- Consider Option 4 (Streamlit + FastAPI) for quick database integration

**Long-term (Production):**
- Option 1 (FastAPI + React/Next.js) for a proper production application
- Provides best balance of reusing existing code while building scalable UI

---

## Deployment Quick Reference

### Current Prototype Sharing
```bash
# Terminal 1
streamlit run ui/app.py

# Terminal 2
ngrok http 8501
```

### Future Hosting Options

| Platform | Cost | Best For |
|----------|------|----------|
| Streamlit Cloud | Free | Prototypes, demos |
| Railway | ~$5/mo | Small production apps |
| Render | ~$7/mo | Small production apps |
| Vercel | Free-$20/mo | Next.js apps |
| AWS/GCP | Variable | Enterprise scale |

---

## Database Schema Considerations

When adding a database, key entities would include:

- **Users** - authentication, preferences
- **Projects** - saved project configurations
- **Scenarios** - saved scenario runs with inputs/outputs
- **Incentive Configurations** - reusable incentive packages

---

## Next Steps (When Ready)

1. Choose architecture based on team skills and timeline
2. Set up database schema
3. Add authentication layer
4. Migrate calculation logic to API endpoints
5. Build/rebuild frontend
6. Deploy to hosting platform

---

*Document created: February 2026*
*Last updated: February 2026*
