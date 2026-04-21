# Figure Production Standard

This is the minimum process for every new static figure or animation in this repo.

## Non-Negotiables

- Start with one teaching target, not a vague topic.
- State what changes and what stays fixed before drawing.
- Use a concrete example whenever the concept is abstract.
- Keep most explanation in the article or caption, not inside the image.
- Reject any figure with visible text overlap, cropped labels, or unreadable collisions.
- Score every shipping figure against the 50-point rubric before push.

## Required Workflow

1. **Teaching brief**
   - Audience level
   - One-sentence lesson
   - Three questions the viewer should be able to answer after seeing it

2. **Concept breakdown**
   - Components
   - Variables and symbols
   - What changes
   - What stays fixed
   - Likely misconceptions

3. **Visual grammar choice**
   - Static for structure, comparison, and scale
   - Animation only when state change is the lesson

4. **Variant pass**
   - Structural option
   - Quantitative option
   - Dynamic option
   - Pick a winner or hybrid before polishing

5. **Render review**
   - Collision check
   - Scale/threshold check
   - Plain-language label check
   - Math alignment check
   - Editorial polish check

6. **Score gate**
   - Use `_notes/figure-quality-rubric.md`
   - Minimum `40/50`
   - No category below `3/5`
   - Any text collision = fail

7. **Animation frame review**
   - Check frame `0`, a middle frame, and the last frame
   - Verify that no annotation collides as values move
   - Verify that the moving element is the lesson, not decoration

8. **Live verification**
   - Rebuild site
   - Push
   - Verify cache-busted asset URLs live

## Fast Repair Order

1. Fix text fit
2. Fix lesson clarity
3. Fix fixed-vs-changing logic
4. Fix quantitative honesty
5. Only then polish spacing, hierarchy, and color

## Delegation Default

When Copilot auth is available, run a small review swarm before finalizing:

- critique lane
- layout lane
- microcopy lane
- animation lane
- red-team lane

Use the swarm to generate options and objections, then merge only the best ideas into the final figure.
