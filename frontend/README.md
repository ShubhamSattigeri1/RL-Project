Artwork Bandit — React Frontend

Quick start

Prerequisites: Node.js (>=18) and npm or pnpm/yarn

From `artwork_bandit/frontend`:

```bash
npm install
npm run dev
```

Open: http://localhost:5173

Notes

- The frontend expects the API running at `http://127.0.0.1:8000`.
- Use the form to enter a `user_id` and `content_id`, click Recommend, then Click/No Click to send feedback.
- To serve the built frontend from FastAPI, build with `npm run build` and copy the `dist` output into a static mount served by the backend.
