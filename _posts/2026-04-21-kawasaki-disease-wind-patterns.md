---
title: "When Kawasaki Disease Started Looking Like a Weather Map"
date: 2026-04-21 00:00:00 -0500
description: "A first-person read of a strange but serious paper on Kawasaki disease, North Pacific wind patterns, and what the evidence can and cannot support."
image:
  path: /assets/img/posts/kawasaki-wind-patterns/hero.png
  alt: January 300 hPa North Pacific wind map with Japan, Hawaii, San Diego, and the P-WIND corridor labeled.
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

The first time I read the Kawasaki disease wind-pattern paper, my reaction was basically: this is either a clever hypothesis or a very polished coincidence.

That kind of paper is usually worth a closer look.

The claim is unusual enough to sound risky at first. Kawasaki disease is a serious childhood illness, and the paper is asking whether its timing in Japan, Hawaii, and San Diego lines up with large-scale wind patterns moving across the North Pacific ([Rodo et al., 2011](#ref-rodo2011)). I would not treat that like a normal regression problem. It is a question about whether a disease pattern and an atmospheric pattern might be part of the same story.

What made the paper worth reading was not the headline. It was the way the authors tried to connect geography, seasonality, and transport without pretending they had already solved the cause.

## The claim sounds weird at first

Kawasaki disease mostly affects young children and can damage the heart and blood vessels. The CDC also notes that the cause is unknown and that it is the leading cause of acquired heart disease in children in the United States ([CDC, 2026](#ref-cdc2026)).

With an open cause, a seasonal and geographic pattern is not just trivia. It is a clue.

I think that is why this paper still feels useful years later. It asks a strange question, but not a random one. If disease activity rises and falls with the seasons in places separated by an ocean, then something in the environment is at least worth testing. The atmosphere is one of the few systems that can move material across that kind of distance on a schedule.

So the paper is not saying, "wind causes Kawasaki disease." It is asking whether a transported trigger, possibly aerosolized, could be part of the mechanism.

## What the paper actually argues

The core argument is narrower than the title sounds.

Rodo and coauthors compared Kawasaki disease timing in Japan, Hawaii, and San Diego with wind fields over the North Pacific. For Japan, they focused on a northwesterly component over the western Pacific. For the trans-Pacific pathway, they defined a Pacific Zonal Wind Index, or P-WIND, along roughly 35 degrees north from 140E to 240E ([Rodo et al., 2011](#ref-rodo2011)).

The important move is not the map itself. It is turning a moving atmospheric field into something seasonal and measurable enough to compare with disease activity.

That was the part I wanted to test for myself.

## The part I could reproduce

I cannot reproduce the clinical case series from the paper from first principles, and I do not need to pretend otherwise. What I can reproduce is the atmospheric side: the North Pacific wind field and the seasonal indices the paper uses to describe it.

For this post, I used NOAA PSL NCEP/NCAR Reanalysis 1 monthly wind data ([NOAA PSL, 2026](#ref-noaa-psl); [Kalnay et al., 1996](#ref-kalnay1996)). Reanalysis is not a station archive. It is a gridded reconstruction of the atmosphere built from observations and a fixed model framework.

I liked it for this question because the paper is about circulation, not just local weather.

That is what the lead map is meant to show: the winter flow is the thing to watch. The paper is interested in whether that pathway changes in a way that lines up with disease timing.

## Turning wind into a number

This is the move that turns a wind field into something I can compare with disease timing.

The atmosphere is continuous. If you want to compare it with disease timing, you need to compress it into something repeatable. The paper does that by using indices.

For the reconstruction in this post, I kept the setup close to the published idea:

- P-WIND: mean 300 hPa east-west wind along 35N, 140E-240E
- Japan NW-WIND: mean 850 hPa northwesterly component over 30-45N, 130-145E
- Period: monthly NOAA reanalysis means from 1996-2006

![Seasonal wind indices derived from NOAA reanalysis data.](/assets/img/posts/kawasaki-wind-patterns/seasonal-wind-indices.png)

*The point of the indices is not that they are perfect. The point is that they turn a large, messy field into something seasonal enough to compare against disease timing.*

In this reconstruction, P-WIND is strongest in February, about 42.94 m/s, and weakest in August, about 5.55 m/s. Japan NW-WIND is strongest in January, about 7.68 m/s, and weakest in August, about 0.24 m/s.

The seasonal gap is the interesting part. The wind pattern is not flat background noise. It changes in a way that could plausibly matter if the disease is responding to something transported through the atmosphere.

## Why the seasonal pattern matters

The monthly wind field makes the seasonal argument easier to see.

![Monthly animation of North Pacific 300 hPa winds.](/assets/img/posts/kawasaki-wind-patterns/monthly-pacific-winds.gif)

*The winter jet strengthens the west-to-east corridor across the Pacific. In summer, that corridor weakens and shifts.*

That does not prove anything by itself. It does, however, explain why the hypothesis is not absurd on its face. Japan, Hawaii, and San Diego are far apart on a map, but they can still sit inside the same broad circulation pattern at the same time of year.

Once you see that, the scientific question changes. It is no longer "can a disease and a weather map look similar?" It becomes "what would move along that path, and can we find evidence for it?"

The paper discusses aerosols and airborne biological material as possibilities, but it does not identify a causal agent.

I would keep that limit in view.

## Where the evidence stops

I like the paper more once I stop asking it to do too much.

It shows a recurring alignment between disease timing and wind structure. I find that interesting, but it is still an association. If I read it carefully, the paper is making a mechanism-shaped hypothesis, not declaring a mechanism as settled fact.

![Workflow figure separating observed case data, reproducible wind data, and the hypothesis.](/assets/img/posts/kawasaki-wind-patterns/evidence-boundary.png)

*The case data are clinical, the wind fields are reproducible, and the transport mechanism remains a hypothesis.*

That boundary is the whole story for me. The paper is strongest when it stays on the evidence side of that line.

I do not think the right takeaway is "wind causes Kawasaki disease." I also do not think the right takeaway is "this is just correlation, so ignore it." The better reading is that the paper identifies a pattern specific enough to justify follow-up work in aerosol sampling, environmental monitoring, and replication across time windows.

## What I would test next

If I were extending this analysis, I would keep the next steps boring and testable:

1. Recompute the wind indices across nearby atmospheric levels.
2. Check whether the seasonal signal survives different averaging windows.
3. Compare the same wind measures against diseases that should not share the pattern.
4. Look for public Kawasaki disease surveillance summaries that can be compared without patient-level data.
5. Test whether the relationship holds outside the original study period.

I would especially care about negative controls. If the same wind index tracks every winter disease, the hypothesis gets weaker. If the pattern stays specific to Kawasaki disease timing and locations, it becomes harder to dismiss.

I trust that kind of result more than a single polished plot. It is also the kind of result that tells you whether the idea is worth carrying forward.

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
