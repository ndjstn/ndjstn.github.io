function getOrCreateId(key) {
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;

  const value = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  window.localStorage.setItem(key, value);
  return value;
}

function performancePayload() {
  const nav = performance.getEntriesByType("navigation")[0];
  if (!nav) return {};

  return {
    type: nav.type,
    domContentLoadedMs: Math.round(nav.domContentLoadedEventEnd),
    loadMs: Math.round(nav.loadEventEnd),
    transferSize: nav.transferSize || 0,
    encodedBodySize: nav.encodedBodySize || 0,
    decodedBodySize: nav.decodedBodySize || 0
  };
}

function connectionPayload() {
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (!connection) return {};

  return {
    effectiveType: connection.effectiveType || "",
    downlink: connection.downlink || 0,
    rtt: connection.rtt || 0,
    saveData: Boolean(connection.saveData)
  };
}

function viewportPayload() {
  return {
    width: window.innerWidth,
    height: window.innerHeight,
    devicePixelRatio: window.devicePixelRatio || 1,
    screenWidth: window.screen ? window.screen.width : 0,
    screenHeight: window.screen ? window.screen.height : 0
  };
}

function trackEvent(name, properties = {}) {
  const payload = {
    site: "linuxoneliners.com",
    name,
    properties,
    at: new Date().toISOString(),
    sessionId: getOrCreateId("lol_session_id"),
    visitorId: getOrCreateId("lol_visitor_id"),
    path: window.location.pathname,
    pageTitle: document.title,
    referrer: document.referrer,
    language: navigator.language || "",
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
    viewport: viewportPayload(),
    connection: connectionPayload(),
    performance: performancePayload()
  };

  window.dispatchEvent(new CustomEvent("linuxoneliners:event", { detail: payload }));

  const body = JSON.stringify(payload);
  const sent = navigator.sendBeacon && navigator.sendBeacon("/api/events", new Blob([body], { type: "application/json" }));

  if (!sent) {
    fetch("/api/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true
    }).catch(() => {});
  }

  if (window.localStorage.getItem("lol_debug_events") === "1") {
    console.info("linuxoneliners:event", payload);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setupSiteSearch() {
  const input = document.querySelector("[data-search-input]");
  const form = document.querySelector("[data-search-form]");
  const results = document.querySelector("[data-search-results]");
  const indexNode = document.getElementById("search-index");

  if (!input || !form || !results || !indexNode) return;

  let items = [];
  try {
    items = JSON.parse(indexNode.textContent).map((item) => ({
      ...item,
      titleText: String(item.title || "").toLowerCase(),
      summaryText: String(item.summary || "").toLowerCase(),
      detailText: String(item.detail || "").toLowerCase(),
      termsText: String(item.terms || "").toLowerCase()
    }));
  } catch {
    return;
  }

  let rankedResults = [];

  function scoreItem(item, terms, query) {
    let score = 0;

    terms.forEach((term) => {
      if (item.titleText === term) score += 140;
      if (item.titleText.startsWith(term)) score += 95;
      if (item.titleText.includes(term)) score += 70;
      if (item.summaryText.includes(term)) score += 36;
      if (item.detailText.includes(term)) score += 28;
      if (item.termsText.includes(term)) score += 18;
    });

    if (item.titleText.includes(query)) score += 35;
    if (item.type === "problem") score += 4;

    return score;
  }

  function renderSearch() {
    const query = input.value.trim().toLowerCase();
    const terms = query.split(/\s+/).filter(Boolean);

    if (query.length < 2 || terms.length === 0) {
      rankedResults = [];
      results.hidden = true;
      results.innerHTML = "";
      return;
    }

    rankedResults = items
      .map((item) => ({ item, score: scoreItem(item, terms, query) }))
      .filter((entry) => entry.score > 0)
      .sort((a, b) => b.score - a.score || a.item.title.localeCompare(b.item.title))
      .slice(0, 8);

    if (rankedResults.length === 0) {
      results.hidden = false;
      results.innerHTML = `
        <p class="search-results-title">No matches</p>
        <p>Try a command, service name, error, or problem area.</p>
      `;
      return;
    }

    const countLabel = rankedResults.length === 1 ? "match" : "matches";
    results.hidden = false;
    results.innerHTML = `
      <p class="search-results-title">Best ${countLabel}</p>
      <div class="search-result-list">
        ${rankedResults.map(({ item }) => `
          <article class="search-result">
            <span class="search-result-meta">${escapeHtml(item.type)} / ${escapeHtml(item.detail)}</span>
            <a href="${escapeHtml(item.url)}" data-track="site_search_result" data-track-label="${escapeHtml(item.type)}:${escapeHtml(item.title)}">${escapeHtml(item.title)}</a>
            <p>${escapeHtml(item.summary)}</p>
          </article>
        `).join("")}
      </div>
    `;
  }

  input.addEventListener("input", renderSearch);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    renderSearch();

    trackEvent("site_search", {
      queryLength: input.value.trim().length,
      resultCount: rankedResults.length
    });

    if (rankedResults[0]) {
      window.location.href = rankedResults[0].item.url;
    }
  });
}

setupSiteSearch();

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy]");
  const tracked = event.target.closest("[data-track]");

  if (tracked) {
    trackEvent(tracked.dataset.track, {
      label: tracked.dataset.trackLabel || "",
      path: window.location.pathname
    });
  }

  if (!button) return;

  try {
    await navigator.clipboard.writeText(button.dataset.copy);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => {
      button.textContent = original;
    }, 1400);
  } catch {
    button.textContent = "Copy failed";
  }
});

window.addEventListener("load", () => {
  trackEvent("page_view", {
    label: document.title
  });
});

const scrollMarks = new Set();
window.addEventListener("scroll", () => {
  const scrollable = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  const percent = Math.min(100, Math.round((window.scrollY / scrollable) * 100));
  [25, 50, 75, 90].forEach((mark) => {
    if (percent >= mark && !scrollMarks.has(mark)) {
      scrollMarks.add(mark);
      trackEvent("scroll_depth", { label: `${mark}`, percent: mark });
    }
  });
}, { passive: true });
