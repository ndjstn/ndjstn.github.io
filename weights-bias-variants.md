---
layout: page
title: Weights/Bias Variants
permalink: /lab/weights-bias-variants/
comments: false
---

<style>
.variant-carousel {
  margin: 1.5rem 0 2rem;
}
.variant-carousel .variant-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.variant-carousel .variant-nav button {
  border: 1px solid #cbd5e1;
  background: #fff;
  color: #0f172a;
  border-radius: 999px;
  padding: 0.55rem 0.9rem;
  font-weight: 600;
  cursor: pointer;
}
.variant-carousel .variant-nav button.active {
  background: #0f172a;
  color: #fff;
  border-color: #0f172a;
}
.variant-carousel .variant-frame {
  border: 1px solid #dbe4ee;
  border-radius: 1rem;
  padding: 1rem;
  background: #fff;
}
.variant-carousel .variant-slide {
  display: none;
}
.variant-carousel .variant-slide.active {
  display: block;
}
.variant-carousel .variant-slide img {
  width: 100%;
  height: auto;
  border-radius: 0.75rem;
}
.variant-carousel .variant-meta {
  margin-top: 1rem;
}
.variant-carousel .variant-meta p {
  margin: 0.35rem 0;
}
.variant-carousel .variant-controls {
  display: flex;
  gap: 0.75rem;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
}
.variant-carousel .variant-controls .arrow-group {
  display: flex;
  gap: 0.75rem;
}
.variant-carousel .variant-controls button {
  border: 1px solid #cbd5e1;
  background: #fff;
  border-radius: 999px;
  padding: 0.5rem 0.9rem;
  cursor: pointer;
  font-weight: 600;
}
.variant-carousel .variant-status {
  color: #475569;
  font-size: 0.95rem;
}
</style>

This is a temporary working page for choosing the clearest figure direction before replacing the article graphic.

<div class="variant-carousel" data-carousel>
  <div class="variant-nav" role="tablist" aria-label="Weights and bias figure variants">
    <button type="button" data-target="0" class="active">A — Structural</button>
    <button type="button" data-target="1">B — Quantitative</button>
    <button type="button" data-target="2">C — Dynamic</button>
  </div>

  <div class="variant-frame">
    <div class="variant-slide active" data-slide="0">
      <img src="/assets/img/posts/neural-network-components/weights-bias-variant-a.png" alt="Variant A structural weights and bias figure." loading="lazy">
      <div class="variant-meta">
        <p><strong>Best for:</strong> first-time readers who need the parts named plainly.</p>
        <p><strong>Shows clearly:</strong> what the pieces are and the order of operations.</p>
        <p><strong>Still weak on:</strong> exact magnitude and sign.</p>
      </div>
    </div>

    <div class="variant-slide" data-slide="1">
      <img src="/assets/img/posts/neural-network-components/weights-bias-variant-b.png" alt="Variant B quantitative weights and bias figure." loading="lazy">
      <div class="variant-meta">
        <p><strong>Best for:</strong> readers who want to see the score get built numerically.</p>
        <p><strong>Shows clearly:</strong> sign, magnitude, contribution, and total score.</p>
        <p><strong>Still weak on:</strong> the fixed-weights / moved-threshold intuition by itself.</p>
      </div>
    </div>

    <div class="variant-slide" data-slide="2">
      <img src="/assets/img/posts/neural-network-components/weights-bias-variant-c.png" alt="Variant C dynamic weights and bias figure." loading="lazy">
      <div class="variant-meta">
        <p><strong>Best for:</strong> people confused about what bias changes.</p>
        <p><strong>Shows clearly:</strong> same weights, sliding threshold, and parallel boundary motion.</p>
        <p><strong>Still weak on:</strong> the actual score buildup unless paired with another panel.</p>
      </div>
    </div>

    <div class="variant-controls">
      <div class="arrow-group">
        <button type="button" data-action="prev">← Prev</button>
        <button type="button" data-action="next">Next →</button>
      </div>
      <div class="variant-status" data-status>Showing A — Structural</div>
    </div>
  </div>
</div>

My default recommendation is still a <strong>hybrid of B + C</strong>:

- use <strong>B</strong> as the main static article image
- keep <strong>C</strong> as the companion animation
- borrow <strong>A</strong>'s plain labeling discipline

<script>
(() => {
  const root = document.querySelector('[data-carousel]');
  if (!root) return;

  const slides = Array.from(root.querySelectorAll('[data-slide]'));
  const tabs = Array.from(root.querySelectorAll('[data-target]'));
  const status = root.querySelector('[data-status]');
  const labels = ['A — Structural', 'B — Quantitative', 'C — Dynamic'];
  let current = 0;

  function render(index) {
    current = (index + slides.length) % slides.length;
    slides.forEach((slide, i) => slide.classList.toggle('active', i === current));
    tabs.forEach((tab, i) => tab.classList.toggle('active', i === current));
    status.textContent = `Showing ${labels[current]}`;
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => render(index));
  });

  root.querySelector('[data-action="prev"]').addEventListener('click', () => render(current - 1));
  root.querySelector('[data-action="next"]').addEventListener('click', () => render(current + 1));
})();
</script>
