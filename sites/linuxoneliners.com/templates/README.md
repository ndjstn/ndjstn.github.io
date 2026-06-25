# Template System

The site is intentionally data-driven.

Core rule:

```text
content/lessons.json -> shared templates -> generated site
```

This lets one feature change update every one-liner page.

## Templates

- `layout.erb`: global HTML shell, nav, heartbeat strip, footer, shared assets
- `index.erb`: homepage, lesson grid, series, curation, exports, vessel status
- `lesson.erb`: every individual one-liner page
- `series.erb`: generated category pages for scaling past a small homepage grid

## Feature Surfaces

To add a feature to every one-liner, change `lesson.erb` once:

- comments section
- SSO prompt
- ad slot
- related lessons
- copy tracking
- demo transcript
- video embed
- affiliate disclosure
- moderation warning

To add a global feature, change `layout.erb` once:

- analytics client
- global nav
- campaign capture
- auth state
- status heartbeat
- CSS/JS references

To add a homepage feature, change `index.erb` once:

- lesson grid
- category sections
- dashboard links
- export links
- conversion funnels

To change category pages, update `series.erb` once. This matters once the site has hundreds of one-liners.

## Future Extraction

When the second or third site uses the same pattern, extract these into a retained operator template:

```text
ops/platform/templates/static-content-site/
```

That retained template should not be sold by default with an individual site. A buyer gets the generated site/source for their site, not the whole site factory.
