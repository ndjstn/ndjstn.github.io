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
