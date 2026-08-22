-- The default `defect` class was seeded alarm-red (#ef4444), which is the wrong register for
-- what it labels. A mask is painted over the photograph at partial opacity for minutes at a
-- time while somebody works; red reads as an error state, competes with nothing else in the
-- palette for attention, and on a dark-field exposure of a metal part it turns muddy brown.
-- Magenta stays legible on metal, on plastic and in dark field, and cannot be mistaken for the
-- teal `signal` accent that draws the selection outline on top of it.
--
-- Only labels still carrying the old default are moved: a colour somebody chose is theirs, and
-- the editor has had a swatch since M9.
UPDATE annotation_label
   SET color = '#c026d3'
 WHERE key = 'defect' AND lower(color) = '#ef4444';
