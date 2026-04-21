# Citation Workflow

This repo now treats citations as first-class content rather than an afterthought.

## Current post workflow

- Use APA-style author-year citations inline in the article body.
- Put a `## References` section at the bottom of the post.
- Link each reference entry to the original DOI or canonical source URL.
- Cite equations in the surrounding sentence if the formulation or algorithm needs attribution.
- Keep the canonical machine-readable entries in `references/library.bib`.

## Why this approach

- It works with the current Chirpy + Jekyll setup with no plugin dependency.
- It renders cleanly on GitHub Pages.
- It keeps the human-readable article and the machine-readable bibliography next to each other.

## Recommended Zotero tandem workflow

- Keep the master library in a private Zotero library or private Zotero group.
- Organize article-specific collections in Zotero.
- Export the relevant collection to BibTeX or BibLaTeX when you update a post.
- Replace or merge those entries into `references/library.bib`.
- Use Zotero for long-term organization, notes, PDFs, and tagging; use this repo for the post-ready subset.

## Future upgrade path

If you want fully automated citations later, the most likely next step is to add `jekyll-scholar` and render citations from `references/library.bib` directly. Because this site already builds through GitHub Actions instead of the restrictive legacy branch builder, that migration is viable when you want it.
