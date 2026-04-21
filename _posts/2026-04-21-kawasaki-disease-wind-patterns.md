---
title: "When Kawasaki Disease Started Looking Like a Weather Map"
date: 2026-04-21 00:00:00 -0500
description: "A paper-inspired data science explainer on Kawasaki disease, North Pacific wind patterns, and the difference between a plausible mechanism and proof."
image:
  path: /assets/img/posts/kawasaki-wind-patterns/hero.png
  alt: North Pacific 300 hPa wind map with Japan, Hawaii, San Diego, and a seasonal wind-index inset.
tags:
  - "Kawasaki Disease"
  - "Climate Data"
  - "NOAA"
  - "Epidemiology"
  - "Scientific Reports"
  - "Data Science"
categories:
  - "Data Science"
---

Most data science examples start with a table.

The Kawasaki disease wind-pattern paper starts with something stranger: a pediatric disease with a seasonal rhythm, a geographic pattern, and no known cause.

That is what makes it interesting. The paper is not a leaderboard problem. It is not "train a model and report accuracy." It is a scientific argument built from timing, geography, and atmospheric circulation.

The 2011 Scientific Reports paper by Rodo and coauthors asked whether Kawasaki disease activity in Japan, Hawaii, and San Diego was associated with large-scale tropospheric winds crossing the North Pacific ([Rodo et al., 2011](#ref-rodo2011)). The paper's most provocative suggestion is that the environmental trigger for Kawasaki disease could be wind-borne.

That sentence needs care.

It does not mean the paper proved what causes Kawasaki disease. It does not mean wind itself causes the disease. It means the authors found repeated alignment between disease timing and atmospheric pathways, strong enough to argue that aerosols or another transported environmental trigger deserved attention.

That is the version worth writing about.

## The disease is real, but the cause is still open

Kawasaki disease is a serious childhood illness. The CDC describes it as a disease that mostly affects children younger than 5 and can damage the heart and blood vessels. It is also the number one cause of acquired heart disease in children in the United States ([CDC, 2026](#ref-cdc2026)).

The cause is not known.

That uncertainty is the whole reason the wind paper matters. If a disease has a stable seasonal rhythm, clusters in time, and appears in distant places with similar timing, it is reasonable to ask whether something outside the patient is moving in a patterned way.

Weather is patterned.

Air masses move.

The atmosphere connects places that look distant on a map.

That is the jump the paper makes: not from "weather causes disease," but from "the timing looks spatially connected" to "maybe we should test whether atmospheric transport is part of the mechanism."

## This is not a Kaggle dataset

The patient data in the paper came from clinical and surveillance sources, not a neat public CSV.

The Japanese series came from nationwide hospital surveys. The paper reports 247,685 cases across 47 prefectures over 1970-2008. The Hawaii analysis used 498 cases from hospital discharge records over 1996-2006. The San Diego series used 749 cases from the UCSD Kawasaki Disease Research Center database over 1994-2008 ([Rodo et al., 2011](#ref-rodo2011)).

That matters for reproducibility.

I cannot honestly reproduce the clinical side from Kaggle. What I can reproduce is the atmospheric side: the North Pacific wind fields and the wind indices the paper uses to describe the seasonal pathway.

For that, I used NOAA PSL's NCEP/NCAR Reanalysis 1 monthly wind data ([NOAA PSL, 2026](#ref-noaa-psl); [Kalnay et al., 1996](#ref-kalnay1996)). Reanalysis is not a weather station spreadsheet. It is a gridded reconstruction of the atmosphere made by combining observations with a fixed weather model and data assimilation system.

The figure at the top of this post is built from those NOAA fields. It shows January winds at 300 hPa, roughly the upper-troposphere level the paper uses when discussing the trans-Pacific pathway to Hawaii and San Diego.

## The paper's move is a data-science move

The clever part is not that the authors drew a weather map.

The clever part is that they converted a messy atmospheric field into indices that could be compared with disease timing.

For Japan, the paper uses a northwesterly wind component. In plain English: are winds over Japan blowing from the north and west, the direction the authors associate with air coming from continental Asia?

For the North Pacific, the paper defines a Pacific Zonal Wind Index, or P-WIND. This is the mean east-west wind along 35 degrees north between 140 degrees east and 240 degrees east. That line runs across the North Pacific, roughly marking the trans-Pacific corridor highlighted in the paper.

The exact implementation below is not a clinical reproduction. It is a reproducible atmospheric echo of the paper's setup:

- P-WIND: mean 300 hPa east-west wind along 35N, 140E-240E
- Japan NW-WIND: mean 850 hPa northwesterly component over 30-45N, 130-145E
- Period: monthly NOAA reanalysis means from 1996-2006

![Seasonal wind indices derived from NOAA reanalysis data.](/assets/img/posts/kawasaki-wind-patterns/seasonal-wind-indices.png)

*The atmospheric indices turn up in the cool season. This figure contains wind data only, not patient records.*

In this reconstruction, the Pacific zonal wind index is strongest in February, about 42.94 m/s, and weakest in August, about 5.55 m/s. The Japan northwesterly component is strongest in January, about 7.68 m/s, and weakest in August, about 0.24 m/s.

That seasonal contrast is the important part. The winds are not constant background noise. The North Pacific pathway changes across the year.

## The animation is the argument

Static maps are useful, but the paper's idea is seasonal. The thing that changes is the circulation.

The animation below shows monthly 300 hPa winds over the North Pacific using the same 1996-2006 NOAA climatology. The map is not trying to show disease cases. It is showing the atmospheric conveyor belt that makes the hypothesis plausible enough to test.

![Monthly animation of North Pacific 300 hPa winds.](/assets/img/posts/kawasaki-wind-patterns/monthly-pacific-winds.gif)

*The winter jet strengthens and opens a clearer west-to-east pathway across the Pacific. In summer, the pattern weakens and shifts.*

That visual makes the paper feel less abstract.

Japan, Hawaii, and San Diego are far apart politically and medically. Atmospherically, they can sit inside connected flow patterns. If a disease has seasonal peaks in those places, and if those peaks repeatedly line up with a seasonal atmospheric pathway, the next scientific question becomes obvious:

What, if anything, is being transported?

The paper discusses aerosols and airborne biological material as possibilities. It does not identify a causal agent. That distinction is not a footnote; it is the central guardrail.

## Association is not causation, but it is not nothing

It is easy to overreact in either direction.

One bad reading says: "The wind causes Kawasaki disease."

Another bad reading says: "Correlation is not causation, so there is nothing here."

Neither is good enough.

The better reading is that the paper builds a mechanistic hypothesis. It combines a clinical pattern with an atmospheric pattern, then says: this alignment is specific enough that aerosol microbiology and environmental sampling should look here.

That is a useful scientific move.

![Workflow figure separating observed case data, reproducible wind data, and the hypothesis.](/assets/img/posts/kawasaki-wind-patterns/evidence-boundary.png)

*The claim has layers. The disease timing is observed in clinical data, the wind fields are reproducible from NOAA, and the transport mechanism remains a hypothesis.*

This is why I like the topic more than a generic Kaggle analysis. A leaderboard dataset often rewards you for treating columns as inputs and a target as output. This paper rewards a different habit: ask whether the data has a mechanism behind it.

If a model finds a pattern but you cannot say how the world could produce that pattern, you may have found a shortcut, a confounder, or noise.

If a pattern lines up with a plausible mechanism, it is not proof. But it is a better reason to investigate.

## What I would do next

If I were turning this into a deeper project, I would not start by fitting a prediction model.

I would start by separating the evidence:

- reproduce the wind fields from NOAA
- rebuild P-WIND and NW-WIND across multiple atmospheric levels
- test whether the seasonal pattern survives different averaging windows
- look for public Kawasaki disease surveillance summaries that can be used without patient-level data
- compare the same wind indices against diseases that should not share the same pattern
- check whether the relationship persists outside the original study period

The negative controls matter. If every winter disease lines up with the same wind index, the hypothesis weakens. If the pattern is specific to Kawasaki disease timing and locations, the hypothesis gets more interesting.

That is where the article should end: not with certainty, but with a sharper question.

## Why this belongs on a data-science site

The lesson is not medical diagnosis. This post is not health advice.

The lesson is how to read a scientific data argument.

The paper starts with a human problem: a serious childhood disease whose cause remains unknown. It then looks for structure across time and space. It converts weather maps into indices. It compares those indices with disease timing. Then it proposes a mechanism that can be tested outside the original analysis.

That is data science at its best.

Not every good analysis ends with a model.

Sometimes the win is a better hypothesis.

## Reproducibility note

The figures in this post were generated from NOAA PSL NCEP/NCAR Reanalysis 1 monthly wind fields for 1996-2006. The script is in the site repository at `scripts/generate_kawasaki_wind_article_images.py`. It accesses NOAA data through OPeNDAP, caches a small regional subset under `.cache/`, and writes the post images under `assets/img/posts/kawasaki-wind-patterns/`.

## References

<div id="ref-rodo2011" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Rodo, X., Ballester, J., Cayan, D., Melish, M. E., Nakamura, Y., Uehara, R., &amp; Burns, J. C. (2011). Association of Kawasaki disease with tropospheric wind patterns. <em>Scientific Reports, 1</em>, Article 152. <a href="https://www.nature.com/articles/srep00152">https://www.nature.com/articles/srep00152</a>
</div>

<div id="ref-cdc2026" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Centers for Disease Control and Prevention. (2026). <em>About Kawasaki disease</em>. <a href="https://www.cdc.gov/kawasaki/about/index.html">https://www.cdc.gov/kawasaki/about/index.html</a>
</div>

<div id="ref-noaa-psl" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
NOAA Physical Sciences Laboratory. (2026). <em>NCEP/NCAR Reanalysis 1</em>. <a href="https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html">https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html</a>
</div>

<div id="ref-kalnay1996" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Kalnay, E., Kanamitsu, M., Kistler, R., Collins, W., Deaven, D., Gandin, L., et al. (1996). The NCEP/NCAR 40-year reanalysis project. <em>Bulletin of the American Meteorological Society, 77</em>(3), 437-471. <a href="https://doi.org/10.1175/1520-0477(1996)077%3C0437:TNYRP%3E2.0.CO;2">https://doi.org/10.1175/1520-0477(1996)077%3C0437:TNYRP%3E2.0.CO;2</a>
</div>
