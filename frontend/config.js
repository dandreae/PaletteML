// Single, obvious place to point the frontend at a backend.
//
// Default (empty string): the frontend calls relative paths
// ("/recommend", "/compare", "/health") on whatever origin served this
// page. That's correct and requires no edits when the frontend is
// served by the same FastAPI app as the API (the default deployment —
// see README's "Deploying to Render" section).
//
// Only set this if the frontend is deployed SEPARATELY from the API
// (e.g. a Render Static Site) — point it at the deployed API's base
// URL, with no trailing slash, e.g.:
//   const PALETTEML_API_BASE_URL = "https://paletteml-api.onrender.com";
const PALETTEML_API_BASE_URL = "";
