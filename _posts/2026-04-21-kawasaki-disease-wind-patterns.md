---
title: "When Kawasaki Disease Started Looking Like a Weather Map"
date: 2026-04-21 00:00:00 -0500
description: "My read of a strange but serious Kawasaki disease paper, the North Pacific wind pattern behind it, and the pathway I would test next."
image:
  path: /assets/img/posts/kawasaki-wind-patterns/hero.png
  alt: January upper-air North Pacific wind map with upstream Asia, Japan, Hawaii, San Diego, a 35N averaging corridor, and a northern route to test.
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

The first time I read the Kawasaki disease wind-pattern paper, I expected the map to be the decorative part. It was not. The thing that stopped me was how specific the setup was: Japan, Hawaii, and San Diego, all showing seasonal Kawasaki disease timing that the authors tied back to North Pacific winds ([Rodo et al., 2011](#ref-rodo2011)). That can be a real lead, or it can be a very good way to fool yourself. I wanted to know which side it leaned toward.

I am not interested in selling the paper as "wind causes Kawasaki disease." That is not what the evidence can carry. The more useful question is smaller and, honestly, more interesting: does the atmospheric side have enough structure that it points to a pathway worth testing?

## The weird part is the specificity

Kawasaki disease mostly affects young children and can damage the heart and blood vessels. The CDC says the cause is not known, and it remains the leading cause of acquired heart disease in children in the United States ([CDC, 2026](#ref-cdc2026)). So when a paper says the timing lines up across places separated by the Pacific, I do not want to hand-wave it away. I also do not want to make it bigger than it is.

The paper's useful claim is not "here is the cause." It is closer to: if a wind-borne trigger exists, the geography should not be random. You would expect a seasonal pathway from continental Asia across Japan and the North Pacific, with Hawaii and southern California sitting downstream during the right months.

That is the part I wanted to rebuild. Not the whole epidemiology paper. Just the wind side.

## What I could actually check

I do not have the clinical case records, so I did not reproduce the disease time series. I rebuilt the atmospheric half using NOAA PSL NCEP/NCAR Reanalysis 1 monthly wind data ([NOAA PSL, 2026](#ref-noaa-psl); [Kalnay et al., 1996](#ref-kalnay1996)). Reanalysis is a gridded reconstruction of the atmosphere, not a set of local weather-station readings, which makes it a good fit for a circulation question.

The lead map is the first sanity check. The red line is not meant to be a literal path from Japan to San Diego. It is the 35N averaging corridor used to summarize west-to-east flow. The more interesting geography is the broader winter setup: upstream Asia into Japan, then trans-Pacific westerlies toward Hawaii and southern California, with a possible northern route around the Aleutians and Gulf of Alaska.

That is already a narrower research path than "something in the air." It gives you places, seasons, and failure modes. If I were turning this into an actual source hunt, I would start with a broad stripe across continental Asia and then try to shrink it.

![Map of source-screening pins at Dunhuang, Dalanzadgad, Lanzhou, Hohhot, Beijing, Shenyang, Changchun, Harbin, Seoul, and Tokyo inside a broader continental Asia wind stripe.](/assets/img/posts/kawasaki-wind-patterns/source-screening-swath.png)

*This is the map I wanted: not a claimed source, but a set of specific pins to sample or cross-check against the wind pathway.*

The first pass would be Dunhuang, Dalanzadgad, Lanzhou, Hohhot, Beijing, Shenyang, Changchun, Harbin, Seoul, and Tokyo. I would not treat those as magic dots on a map. I would treat them as places to pull soils, dust, crop residue, fungi and yeasts, toxins, and fine aerosols into the same comparison frame.

I would treat that stripe as a sampling plan. The desert regions matter because Asian dust-source areas can loft bacterial and fungal bioaerosols ([Maki et al., 2019](#ref-maki2019)). The North China and northeast China pieces matter because later flight sampling over Japan, designed around suspected source-region winds, found a strong fungal signal including Candida, while still stopping short of naming a causal agent ([Rodo et al., 2014](#ref-rodo2014)). The agricultural and soil regions matter because toxins, plant decay, fungi, yeasts, and dust-attached cofactors are all different enough that I would not want one generic "upstream Asia" bucket.

## Turning wind into a number

To compare wind with disease timing, you need to turn the map into something repeatable. The paper does that with wind indices. Rodo and coauthors define P-WIND as the mean zonal wind along 35N from 140E to 240E; their seasonal case comparison shows it at the surface, with similar behavior reported in the middle and upper troposphere ([Rodo et al., 2011](#ref-rodo2011)).

For this post I used the same corridor idea, but the map and animation use the 300 hPa upper-air field because that makes the Pacific jet and possible transport corridor easier to see:

1. P-WIND here: mean 300 hPa east-west wind along 35N, 140E-240E.
2. Japan NW-WIND here: 850 hPa wind projected onto a northwest/southeast axis over 30-45N, 130-145E.
3. Period: 1996-2006 monthly climatology from NOAA reanalysis.

![Seasonal wind indices showing cool-season peaks in the Pacific corridor and Japan northwesterly flow.](/assets/img/posts/kawasaki-wind-patterns/seasonal-wind-indices.png)

*This is the compression step. A big moving atmosphere becomes two monthly signals I can actually compare and try to break.*

In this reconstruction, P-WIND peaks in February at about 42.94 m/s and bottoms out in August at about 5.55 m/s. Japan NW-WIND peaks in January at about 7.68 m/s and bottoms out in August at about 0.24 m/s. The exact numbers are less important than the shape. The wind is not a flat background. It has a seasonal pulse.

## The pathway question

The monthly wind field is where the idea becomes easier to argue with. The moving dots below are not a simulation of the causal agent. They are tracer pulses pushed by the monthly 300 hPa wind field, which makes the open corridor easier to see and keeps the upstream inspection areas on the map.

![Animation of tracer pulses moving through monthly North Pacific upper-air wind fields, with Mongolia, the Gobi Desert, north China, Japan, Hawaii, and San Diego labeled.](/assets/img/posts/kawasaki-wind-patterns/wind-pathway-particles.gif)

*The gold areas are where I would start looking upstream. The moving dots follow the wind field so the winter pathway reads like motion instead of a static weather map.*

I would not call that a mechanism. I would call it a better target. If there is a transported trigger, I would start looking upstream of Japan and then along the winter Pacific corridor toward Hawaii and southern California. I would also keep the northern Pacific branch in play, because the paper itself treats the atmospheric path as more than one clean line across the ocean.

The thing to look for is not just "aerosols" in the abstract. It is a candidate exposure that appears in the right upstream region, survives the relevant atmospheric conditions, and shows up when the pathway is open. That is a much harder claim, but at least it is a claim you can test.

## Where the guess begins

This is where I would draw the line. The paper shows a recurring alignment between disease timing and wind structure. My rebuild shows that the atmospheric part of that story is real enough to reproduce. None of that identifies the agent.

![Research map showing upstream inspection regions, the Pacific wind corridor, observed downstream locations, and a warning that the causal agent is not shown.](/assets/img/posts/kawasaki-wind-patterns/research-boundary-map.png)

*This is the useful output: a search area, not a proof chain.*

I do not buy "wind causes Kawasaki disease." I do buy that the wind pattern is specific enough to be a research lead. That distinction matters. If the map gets treated like proof, the paper becomes too neat. If it gets dismissed as "just correlation," the best part of the paper gets lost.

## What I would do next

I would try to break the pathway before I tried to defend it.

1. Run trajectory or back-trajectory tests from peak months in Japan, Hawaii, and San Diego.
2. Recompute the indices across nearby pressure levels, windows, and the northern route.
3. Use negative controls: diseases, places, or seasons that should not share the same wind signal.
4. If the signal survives that, compare it with public surveillance summaries and targeted aerosol or environmental sampling.

Negative controls are the key piece for me. If the same index lights up every winter illness, the idea gets weak fast. If it stays specific to Kawasaki disease timing and the places the paper points to, then the map is doing more than looking interesting. It is narrowing the search.

## Reproducibility note

The figures in this post were generated from NOAA PSL NCEP/NCAR Reanalysis 1 monthly wind fields for 1996-2006. The script is in the site repository at `scripts/generate_kawasaki_wind_article_images.py`. It accesses NOAA data through OPeNDAP, caches a regional subset under `.cache/`, and writes the post images under `assets/img/posts/kawasaki-wind-patterns/`. The Python package versions I used are recorded in `requirements-kawasaki-wind.txt`.

## References

<div id="ref-rodo2011" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Rodo, X., Ballester, J., Cayan, D., Melish, M. E., Nakamura, Y., Uehara, R., &amp; Burns, J. C. (2011). Association of Kawasaki disease with tropospheric wind patterns. <em>Scientific Reports, 1</em>, Article 152. <a href="https://www.nature.com/articles/srep00152">https://www.nature.com/articles/srep00152</a>
</div>

<div id="ref-rodo2014" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Rodo, X., Curcoll, R., Robinson, M., Ballester, J., Burns, J. C., Cayan, D. R., et al. (2014). Tropospheric winds from northeastern China carry the etiologic agent of Kawasaki disease from its source to Japan. <em>Proceedings of the National Academy of Sciences, 111</em>(22), 7952-7957. <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4050536/">https://pmc.ncbi.nlm.nih.gov/articles/PMC4050536/</a>
</div>

<div id="ref-maki2019" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Maki, T., Hara, K., Kobayashi, F., Kurosaki, Y., Kakikawa, M., Matsuki, A., et al. (2019). Vertical distributions of airborne microorganisms over Asian dust source region of Taklimakan and Gobi Desert. <em>Atmospheric Environment, 214</em>, 116848. <a href="https://doi.org/10.1016/j.atmosenv.2019.116848">https://doi.org/10.1016/j.atmosenv.2019.116848</a>
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
