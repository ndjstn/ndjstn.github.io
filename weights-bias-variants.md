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
.variant-carousel .variant-controls {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-start;
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
</style>

Flip through the variants and tell me what reads best and what feels off.

<div class="variant-carousel" data-carousel>
  <div class="variant-nav" role="tablist" aria-label="Weights and bias figure variants">
    <button type="button" data-target="0" class="active">A — Structural</button>
    <button type="button" data-target="1">B — Quantitative</button>
    <button type="button" data-target="2">C — Dynamic</button>
  </div>

  <div class="variant-frame">
    <div class="variant-slide active" data-slide="0">
      <img src="/assets/img/posts/neural-network-components/weights-bias-variant-a.png" alt="Variant A structural weights and bias figure." loading="lazy">
    </div>

    <div class="variant-slide" data-slide="1">
      <img src="/assets/img/posts/neural-network-components/weights-bias-variant-b.png" alt="Variant B quantitative weights and bias figure." loading="lazy">
    </div>

    <div class="variant-slide" data-slide="2">
      <img src="/assets/img/posts/neural-network-components/weights-bias-variant-c.png" alt="Variant C dynamic weights and bias figure." loading="lazy">
    </div>

    <div class="variant-controls">
      <div class="arrow-group">
        <button type="button" data-action="prev">← Prev</button>
        <button type="button" data-action="next">Next →</button>
      </div>
    </div>
  </div>
</div>
