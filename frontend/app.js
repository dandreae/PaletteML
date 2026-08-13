// PaletteML frontend logic. Plain JS, no framework, no build step.
//
// Structured in two layers on purpose:
//   1. Pure functions (validation, request building, response
//      formatting, error-message extraction) — no DOM access, fully
//      unit-testable from Node (see app.test.js) and reused by the
//      DOM layer below.
//   2. DOM wiring — reads/writes the page, calls the API, calls the
//      pure functions to decide what to show.
//
// API base URL comes from config.js's PALETTEML_API_BASE_URL (loaded
// before this file — see index.html) so it never needs editing here.

// ---------------------------------------------------------------------
// 1. Pure logic (exported at the bottom for Node's test runner)
// ---------------------------------------------------------------------

const HEX_RE = /^#?[0-9a-fA-F]{6}$/;

/** Is `value` a valid 6-digit hex color, with or without a leading "#"? */
function isValidHex(value) {
  return typeof value === "string" && HEX_RE.test(value.trim());
}

/** Normalize to "#rrggbb" lowercase, or null if invalid. */
function normalizeHex(value) {
  if (!isValidHex(value)) return null;
  const trimmed = value.trim();
  const withHash = trimmed.startsWith("#") ? trimmed : `#${trimmed}`;
  return withHash.toLowerCase();
}

const MAX_SEED_COLORS = 10; // mirrors the API's RecommendRequest.colors max_length
const DEFAULT_TOP_N = 5;
const MIN_TOP_N = 1;
const MAX_TOP_N = 20; // mirrors the API's RecommendRequest.top_n le=20

/** Clamp a top_n value into the range the API accepts. */
function clampTopN(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return DEFAULT_TOP_N;
  return Math.min(MAX_TOP_N, Math.max(MIN_TOP_N, n));
}

/** Build the JSON body /recommend and /compare both expect. */
function buildRequestBody(seedColors, topN) {
  return { colors: seedColors, top_n: clampTopN(topN) };
}

function formatScore(score) {
  return Number(score).toFixed(3);
}

/** [{hex, score}] from the API -> [{hex, scoreText}] ready to render. */
function formatRecommendations(recommendations) {
  if (!Array.isArray(recommendations)) return [];
  return recommendations.map((r) => ({ hex: r.hex, scoreText: formatScore(r.score) }));
}

/**
 * Turn a failed fetch (HTTP status + parsed JSON body, if any) into a
 * short, friendly message — never surfaces raw JSON to the user.
 * `status === 0` signals a network-level failure (no response at all).
 */
function extractErrorMessage(status, body) {
  if (status === 0) {
    return "Could not reach the server. Check your connection and try again.";
  }
  if (body && Array.isArray(body.detail) && body.detail.length > 0) {
    // FastAPI/Pydantic validation error shape: {"detail": [{"msg": "...", ...}]}
    const first = body.detail[0];
    if (first && typeof first.msg === "string") return first.msg;
  }
  if (body && typeof body.detail === "string") {
    return body.detail;
  }
  if (status === 422) {
    return "That input isn't valid — check the hex colors and try again.";
  }
  if (status >= 500) {
    return "The server hit an error generating recommendations. Please try again.";
  }
  return `Request failed (HTTP ${status}). Please try again.`;
}

// ---------------------------------------------------------------------
// 2. DOM wiring — skipped entirely under Node (no `window` there),
//    so app.test.js can require() this file for the pure functions
//    above without a browser/DOM present.
// ---------------------------------------------------------------------

if (typeof window !== "undefined") {
  (function initPaletteMLApp() {
    const colorPicker = document.getElementById("color-picker");
    const hexInput = document.getElementById("hex-input");
    const hexError = document.getElementById("hex-error");
    const addSeedBtn = document.getElementById("add-seed-btn");
    const seedChips = document.getElementById("seed-chips");
    const topNInput = document.getElementById("top-n");
    const compareToggle = document.getElementById("compare-toggle");
    const generateBtn = document.getElementById("generate-btn");
    const resultsPanel = document.getElementById("results-panel");

    const apiBase = (typeof PALETTEML_API_BASE_URL === "string" && PALETTEML_API_BASE_URL) || "";

    let seedColors = [];

    // --- seed color management ---

    function showHexError(message) {
      hexError.textContent = message;
    }

    function clearHexError() {
      hexError.textContent = "";
    }

    function addSeedColor(rawValue) {
      const hex = normalizeHex(rawValue);
      if (!hex) {
        showHexError(`"${rawValue || ""}" isn't a valid hex color — try something like #b23a2f.`);
        return false;
      }
      if (seedColors.includes(hex)) {
        showHexError(`${hex} is already added.`);
        return false;
      }
      if (seedColors.length >= MAX_SEED_COLORS) {
        showHexError(`You can add up to ${MAX_SEED_COLORS} seed colors.`);
        return false;
      }
      seedColors.push(hex);
      clearHexError();
      renderSeedChips();
      return true;
    }

    function removeSeedColor(hex) {
      seedColors = seedColors.filter((c) => c !== hex);
      renderSeedChips();
    }

    function renderSeedChips() {
      seedChips.innerHTML = "";
      for (const hex of seedColors) {
        const chip = document.createElement("span");
        chip.className = "seed-chip";

        const swatch = document.createElement("span");
        swatch.className = "seed-chip__swatch";
        swatch.style.setProperty("--chip-color", hex);

        const label = document.createElement("span");
        label.className = "seed-chip__label";
        label.textContent = hex;

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "seed-chip__remove";
        remove.setAttribute("aria-label", `Remove ${hex}`);
        remove.textContent = "×"; // ×
        remove.addEventListener("click", () => removeSeedColor(hex));

        chip.append(swatch, label, remove);
        seedChips.appendChild(chip);
      }
    }

    // keep the color picker and hex text field in sync with each other
    colorPicker.addEventListener("input", () => {
      hexInput.value = colorPicker.value;
    });
    hexInput.addEventListener("input", () => {
      const hex = normalizeHex(hexInput.value);
      if (hex) colorPicker.value = hex;
    });
    hexInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        if (addSeedColor(hexInput.value)) hexInput.value = "";
      }
    });
    addSeedBtn.addEventListener("click", () => {
      if (addSeedColor(hexInput.value || colorPicker.value)) hexInput.value = "";
    });

    // --- API calls ---

    async function fetchJson(path, body, { onSlow } = {}) {
      const url = `${apiBase}${path}`;
      let slowTimer = null;
      if (onSlow) slowTimer = setTimeout(onSlow, 4000);

      let response;
      try {
        response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (networkError) {
        if (slowTimer) clearTimeout(slowTimer);
        throw new Error(extractErrorMessage(0, null));
      }
      if (slowTimer) clearTimeout(slowTimer);

      let parsedBody = null;
      try {
        parsedBody = await response.json();
      } catch {
        // empty/non-JSON body — extractErrorMessage below handles null fine
      }

      if (!response.ok) {
        throw new Error(extractErrorMessage(response.status, parsedBody));
      }
      return parsedBody;
    }

    // --- rendering ---

    function setLoading(message) {
      resultsPanel.innerHTML = "";
      const status = document.createElement("p");
      status.className = "status-message status-message--loading";
      status.textContent = message;
      resultsPanel.appendChild(status);
    }

    function renderError(message) {
      resultsPanel.innerHTML = "";
      const status = document.createElement("p");
      status.className = "status-message status-message--error";
      status.textContent = message;
      resultsPanel.appendChild(status);
    }

    function buildEmptyMessage(text) {
      const p = document.createElement("p");
      p.className = "status-message";
      p.textContent = text;
      return p;
    }

    function copyToClipboard(text, onDone) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(onDone).catch(() => onDone(false));
      } else {
        onDone(false);
      }
    }

    function buildSwatchCard(rec) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "swatch-card";
      card.setAttribute("aria-label", `Copy ${rec.hex}`);

      const color = document.createElement("span");
      color.className = "swatch-card__color";
      color.style.setProperty("--swatch-color", rec.hex);

      const hexLabel = document.createElement("span");
      hexLabel.className = "swatch-card__hex";
      hexLabel.textContent = rec.hex;

      const scoreLabel = document.createElement("span");
      scoreLabel.className = "swatch-card__score";
      scoreLabel.textContent = `score ${formatScore(rec.score)}`;

      card.append(color, hexLabel, scoreLabel);

      card.addEventListener("click", () => {
        copyToClipboard(rec.hex, (ok) => {
          const original = hexLabel.textContent;
          hexLabel.textContent = ok === false ? "Copy failed" : "Copied!";
          card.classList.add("swatch-card--copied");
          setTimeout(() => {
            hexLabel.textContent = original;
            card.classList.remove("swatch-card--copied");
          }, 1200);
        });
      });

      return card;
    }

    function renderSingleResults(data) {
      resultsPanel.innerHTML = "";
      const recs = (data && data.recommendations) || [];

      if (recs.length === 0) {
        resultsPanel.appendChild(
          buildEmptyMessage("No strong recommendations for this color yet — try a different seed.")
        );
        return;
      }

      const grid = document.createElement("div");
      grid.className = "swatch-grid";
      for (const rec of recs) grid.appendChild(buildSwatchCard(rec));
      resultsPanel.appendChild(grid);
    }

    const METHOD_LABELS = {
      svd: "SVD embedding",
      co_occurrence: "Direct PPMI / co-occurrence",
      popularity: "Popularity baseline",
    };
    const METHOD_ORDER = ["svd", "co_occurrence", "popularity"];

    function renderCompareResults(data) {
      resultsPanel.innerHTML = "";
      const results = (data && data.results) || {};

      const compareWrap = document.createElement("div");
      compareWrap.className = "compare-grid";

      for (const method of METHOD_ORDER) {
        const recs = results[method] || [];
        const column = document.createElement("section");
        column.className = `compare-column compare-column--${method}`;

        const heading = document.createElement("h3");
        heading.className = "compare-column__label";
        heading.textContent = METHOD_LABELS[method] || method;
        column.appendChild(heading);

        if (recs.length === 0) {
          column.appendChild(buildEmptyMessage("No recommendations."));
        } else {
          const grid = document.createElement("div");
          grid.className = "swatch-grid swatch-grid--compact";
          for (const rec of recs) grid.appendChild(buildSwatchCard(rec));
          column.appendChild(grid);
        }

        compareWrap.appendChild(column);
      }

      resultsPanel.appendChild(compareWrap);
    }

    // --- generate ---

    async function handleGenerate() {
      if (seedColors.length === 0) {
        const fallback = hexInput.value || colorPicker.value;
        if (!addSeedColor(fallback)) return; // error already shown inline
      }

      const topN = clampTopN(topNInput.value);
      const isCompare = compareToggle.checked;
      const body = buildRequestBody(seedColors, topN);
      const onSlow = () =>
        setLoading("Waking up the model… Render's free tier can take up to a minute after being idle.");

      setLoading(isCompare ? "Comparing models…" : "Generating recommendations…");
      generateBtn.disabled = true;
      try {
        if (isCompare) {
          const data = await fetchJson("/compare", body, { onSlow });
          renderCompareResults(data);
        } else {
          const data = await fetchJson("/recommend", body, { onSlow });
          renderSingleResults(data);
        }
      } catch (err) {
        renderError(err.message);
      } finally {
        generateBtn.disabled = false;
      }
    }

    generateBtn.addEventListener("click", handleGenerate);

    // seed the UI with one default color so the page never looks empty
    addSeedColor(colorPicker.value);
  })();
}

// ---------------------------------------------------------------------
// Node test-runner export (no-op in the browser — `module` is undefined there)
// ---------------------------------------------------------------------
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    isValidHex,
    normalizeHex,
    clampTopN,
    buildRequestBody,
    formatScore,
    formatRecommendations,
    extractErrorMessage,
    MAX_SEED_COLORS,
    DEFAULT_TOP_N,
    MIN_TOP_N,
    MAX_TOP_N,
  };
}
