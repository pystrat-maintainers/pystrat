# Changelog

## 2.2.1

### Fixed
- enforce integer values for `plot_swatch` and in `pystrat.Style`

### Changed
- remove x-axis styling in `pystrat.Style.plot_legend()`


## 2.2.0

### Added
- `pystrat.Style.plot_legend(annotations_loc='right')`, `annotations_loc=` parameter controls where annotations plot in the Style legend
- more annotations
- `pystrat.Section.plot_data_attribute(clean=True)`, `clean=` parameter helps with styling of data attribute plotting

## 2.1.0

### Added
- `pystrat` now ships with a small (but growing) library of default annotations for plotting alongside sections
  - `pystrat.Style(annotations=)` can be initialized with a list containing a subset of the default annotations
- `pystrat.Style.plot_default_annotations()` provides visualization of the default annotations

## 2.0.1

### Fixed
- swatches now part of package data and will work after pip install

## 2.0.0

### Added
- new `Fence` class for combining Sections
  - this class is currently capable of plotting a Fence diagram
  - traces between correlative units can be plotted
  - distances between sections can be scaled and labeled
  - data attributes can be plotted alongside sections that have them
- `Section`
  - Added class method `Section.shift_heights()` to Section to shift stratigraphic heights by a fixed amount (also shifts data attribute heights)
  - Sections can now have names
  - Sections can have (multiple) unit labels in boxes on the left, as is commonly done for labeling of formations, groups, etc.
  - Sections can have annotations plot at specified heights to indicate observations
  - `Section.get_units()` returns the units at requested stratigraphic heights
- `Style`
  - USGS swatches can be defined as an optional component
  - Annotations can be defined as an optional component
  - `height_scaling_factor` and `width_inches` are no longer parameters of `Style` because, as implemented, they introduced unncecessary complciations with plotting of swatches and `Axes` manipulation. Re-introduction of a scaling factor might be worthwhile, but the new `Fence` class ensures that sections plotted alongside each other will share identical vertical scales, which should account for the majority of cases where this would be desired. Otherwise, a user can simply create `Axes` that share the y-axis, which is very straightforward in `matplotlib`.
- misc
  - `section_style_compatibility()` is now `Section.style_compatibility()`, and style checking occurs before plotting

### Changed

#### Styling
Colors and widths are no longer separate stylistic decisions tied to potentially separate attributes. Rather, styling is now tied to an single facies attribute, and every facies has its own color and width. This change reflects the opinion that stratigraphic sections are visualizations of subjectively determined phenomena of interest, which inevitably resolve at a "facies" scale. Any stylistic considerations should then reflect the facies interpretation directly, rather than distributing facies descriptors into multiple, less interpretable attributes. This approach also simplifies the code by only permitting styling to vary along a single axis.

#### Plotting
- Plotting is less prescriptive now. Decisions about subplots and figure/axis dimensions are made externally to `pystrat`, and the user has more flexibility in supplying axes to plot into.
- Plotting a section is now a class method (`plot_stratigraphy()` &rarr; `Section.plot()`)
- Section plotting no longer starts strictly at 0; instead the base of the section is taken as the first entry of `Section.base_heights` (which may be non-zero if the user has used `Section.shift_heights()`)
- Plotting the legend for a style is more flexible