"""SVG export helpers for matplotlib pystrat figures."""

from __future__ import annotations

import copy
import math
import re
import tempfile
from importlib import resources
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
SWATCH_GID_RE = re.compile(r"^pystrat-swatch-(\d+)(?:-.+)?$")
ANNOTATION_GID_RE = re.compile(r"^pystrat-annotation-(.+)$")
__all__ = [
    "available_svg_swatches",
    "inject_annotation_previews",
    "inject_swatch_patterns",
    "inject_swatch_previews",
    "make_annotation_gid",
    "make_swatch_gid",
    "plot_svg_swatch_catalog",
    "savefig_svg",
    "svg_file_viewbox_size",
]

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def svg_tag(tag):
    return f"{{{SVG_NS}}}{tag}"


def parse_style(style_text):
    style = {}
    if not style_text:
        return style
    for item in style_text.split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        style[key.strip()] = value.strip()
    return style


def format_style(style):
    return "; ".join(f"{key}: {value}" for key, value in style.items())


def swatch_svg_path(swatch_code):
    swatch_code = int(swatch_code)
    swatch_path = resources.files("pystrat").joinpath("swatches_svg", f"{swatch_code}.svg")
    if not swatch_path.is_file():
        raise ValueError(f"SVG swatch {swatch_code} not found.")
    return swatch_path


def svg_viewbox_parts(svg_root):
    view_box = svg_root.attrib.get("viewBox")
    if view_box:
        parts = [float(x) for x in view_box.replace(",", " ").split()]
        if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
            return parts[0], parts[1], parts[2], parts[3], view_box
    width = float(re.sub(r"[^0-9.]+", "", svg_root.attrib.get("width", "54.1")))
    height = float(re.sub(r"[^0-9.]+", "", svg_root.attrib.get("height", "54.1")))
    return 0.0, 0.0, width, height, f"0 0 {width:g} {height:g}"


def svg_viewbox_size(svg_root):
    view_x, view_y, width, height, view_box = svg_viewbox_parts(svg_root)
    return width, height, view_box


def svg_file_viewbox_size(svg_path):
    svg_root = ET.parse(svg_path).getroot()
    return svg_viewbox_size(svg_root)


def namespace_svg_asset(svg_root, prefix):
    id_prefix = f"{prefix}-id-"
    class_styles = {}
    id_map = {}

    for elem in svg_root.iter():
        if elem.tag != svg_tag("style"):
            continue
        for class_name, declarations in re.findall(
            r"\.([A-Za-z_][A-Za-z0-9_-]*)\s*\{([^}]*)\}", elem.text or ""
        ):
            class_styles[class_name] = declarations.strip().rstrip(";")

    for elem in svg_root.iter():
        elem_id = elem.attrib.get("id")
        if elem_id:
            id_map[elem_id] = id_prefix + elem_id

    def replace_ids(text):
        for old_id, new_id in id_map.items():
            text = text.replace(f"url(#{old_id})", f"url(#{new_id})")
            text = text.replace(f"#{old_id}", f"#{new_id}")
        return text

    for elem in svg_root.iter():
        class_attr = elem.attrib.get("class")
        if class_attr:
            declarations = []
            for class_name in class_attr.split():
                class_style = class_styles.get(class_name)
                if class_style:
                    declarations.append(replace_ids(class_style))
            existing_style = elem.attrib.get("style")
            if existing_style:
                declarations.append(replace_ids(existing_style.strip().rstrip(";")))
            if declarations:
                elem.set("style", "; ".join(declarations))
            elem.attrib.pop("class", None)

        elem_id = elem.attrib.get("id")
        if elem_id in id_map:
            elem.set("id", id_map[elem_id])

        for attr, value in list(elem.attrib.items()):
            if isinstance(value, str) and ("url(#" in value or value.startswith("#")):
                elem.set(attr, replace_ids(value))


def namespace_swatch(svg_root, swatch_code, suffix=""):
    prefix = f"sw{swatch_code}-{suffix}" if suffix else f"sw{swatch_code}"
    namespace_svg_asset(svg_root, prefix)


def shape_bbox(shape):
    if shape.tag == svg_tag("rect"):
        x = float(shape.attrib.get("x", "0"))
        y = float(shape.attrib.get("y", "0"))
        width = float(shape.attrib.get("width", "0"))
        height = float(shape.attrib.get("height", "0"))
        return x, y, width, height

    path_data = shape.attrib.get("d", "")
    values = [
        float(value)
        for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path_data)
    ]
    if len(values) < 4:
        return None
    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def make_swatch_gid(swatch_code, unique_id=None):
    if unique_id is None:
        return f"pystrat-swatch-{int(swatch_code)}"
    return f"pystrat-swatch-{int(swatch_code)}-{unique_id}"


def make_annotation_gid(unique_id):
    return f"pystrat-annotation-{unique_id}"


def make_pattern_def(swatch_code, pattern_id, tile_width_pt, pattern_x=None, pattern_y=None):
    swatch_tree = ET.parse(swatch_svg_path(swatch_code))
    swatch_root = swatch_tree.getroot()
    view_w, view_h, view_box = svg_viewbox_size(swatch_root)
    namespace_swatch(swatch_root, swatch_code, pattern_id)

    tile_height_pt = tile_width_pt * view_h / view_w
    attrs = {
        "id": pattern_id,
        "patternUnits": "userSpaceOnUse",
        "width": f"{tile_width_pt:g}",
        "height": f"{tile_height_pt:g}",
        "viewBox": view_box,
    }
    if pattern_x is not None:
        attrs["x"] = f"{pattern_x:g}"
    if pattern_y is not None:
        attrs["y"] = f"{pattern_y:g}"

    pattern = ET.Element(svg_tag("pattern"), attrs)
    for child in list(swatch_root):
        if child.tag == svg_tag("style"):
            continue
        pattern.append(copy.deepcopy(child))
    return pattern


def inject_swatch_patterns(
    svg_path,
    output_path=None,
    tile_width_pt=108.0,
    center_patterns=True,
):
    svg_path = Path(svg_path)
    output_path = Path(output_path) if output_path is not None else svg_path

    tree = ET.parse(svg_path)
    root = tree.getroot()
    defs = root.find(svg_tag("defs"))
    if defs is None:
        defs = ET.Element(svg_tag("defs"))
        root.insert(0, defs)

    pattern_specs = []
    for group in root.iter(svg_tag("g")):
        match = SWATCH_GID_RE.match(group.attrib.get("id", ""))
        if not match:
            continue
        code = int(match.group(1))
        shapes = list(group.iter(svg_tag("path"))) + list(group.iter(svg_tag("rect")))
        for shape in shapes:
            pattern_id = f"pystrat-pattern-{code}-{len(pattern_specs)}"
            pattern_x = pattern_y = None
            bbox = shape_bbox(shape)
            if center_patterns and bbox is not None:
                x, y, width, height = bbox
                pattern_x = x + width / 2 - tile_width_pt / 2
                pattern_y = y + height / 2 - tile_width_pt / 2
            style = parse_style(shape.attrib.get("style"))
            style["fill"] = f"url(#{pattern_id})"
            style.pop("fill-opacity", None)
            shape.set("style", format_style(style))
            pattern_specs.append((code, pattern_id, pattern_x, pattern_y))

    for code, pattern_id, pattern_x, pattern_y in pattern_specs:
        defs.append(
            make_pattern_def(
                code,
                pattern_id,
                tile_width_pt,
                pattern_x=pattern_x,
                pattern_y=pattern_y,
            )
        )

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def inject_swatch_previews(svg_path, output_path=None):
    svg_path = Path(svg_path)
    output_path = Path(output_path) if output_path is not None else svg_path

    tree = ET.parse(svg_path)
    root = tree.getroot()

    for preview_index, group in enumerate(root.iter(svg_tag("g"))):
        match = SWATCH_GID_RE.match(group.attrib.get("id", ""))
        if not match:
            continue

        code = int(match.group(1))
        shapes = list(group.iter(svg_tag("path"))) + list(group.iter(svg_tag("rect")))
        if not shapes:
            continue
        bbox = shape_bbox(shapes[0])
        if bbox is None:
            continue
        x, y, width, height = bbox

        swatch_tree = ET.parse(swatch_svg_path(code))
        swatch_root = swatch_tree.getroot()
        view_x, view_y, view_w, view_h, view_box = svg_viewbox_parts(swatch_root)
        namespace_swatch(swatch_root, code, f"preview-{preview_index}")

        scale_x = width / view_w
        scale_y = height / view_h
        translate_x = x - view_x * scale_x
        translate_y = y - view_y * scale_y
        preview = ET.Element(
            svg_tag("g"),
            {"transform": f"translate({translate_x:g} {translate_y:g}) scale({scale_x:g} {scale_y:g})"},
        )
        for child in list(swatch_root):
            if child.tag == svg_tag("style"):
                continue
            preview.append(copy.deepcopy(child))

        group.insert(0, preview)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def inject_annotation_previews(svg_path, annotation_paths, output_path=None):
    svg_path = Path(svg_path)
    output_path = Path(output_path) if output_path is not None else svg_path

    tree = ET.parse(svg_path)
    root = tree.getroot()

    for preview_index, group in enumerate(root.iter(svg_tag("g"))):
        group_id = group.attrib.get("id", "")
        if not ANNOTATION_GID_RE.match(group_id):
            continue

        annotation_path = annotation_paths.get(group_id)
        if annotation_path is None:
            continue

        shapes = list(group.iter(svg_tag("path"))) + list(group.iter(svg_tag("rect")))
        if not shapes:
            continue
        bbox = shape_bbox(shapes[0])
        if bbox is None:
            continue
        x, y, width, height = bbox

        annotation_tree = ET.parse(annotation_path)
        annotation_root = annotation_tree.getroot()
        view_x, view_y, view_w, view_h, view_box = svg_viewbox_parts(annotation_root)
        namespace_svg_asset(annotation_root, f"ann-preview-{preview_index}")

        scale_x = width / view_w
        scale_y = height / view_h
        translate_x = x - view_x * scale_x
        translate_y = y - view_y * scale_y
        preview = ET.Element(
            svg_tag("g"),
            {"transform": f"translate({translate_x:g} {translate_y:g}) scale({scale_x:g} {scale_y:g})"},
        )
        for child in list(annotation_root):
            if child.tag == svg_tag("style"):
                continue
            preview.append(copy.deepcopy(child))

        group.insert(0, preview)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def collect_annotation_svgs(fig):
    annotation_paths = {}
    for artist in fig.findobj():
        get_gid = getattr(artist, "get_gid", None)
        if get_gid is None:
            continue
        gid = artist.get_gid()
        if not gid or not ANNOTATION_GID_RE.match(gid):
            continue
        annotation_path = getattr(artist, "_pystrat_annotation_svg_path", None)
        if annotation_path:
            annotation_paths[gid] = Path(annotation_path)
    return annotation_paths


def savefig_svg(fig, output_path, tile_width_pt=108.0, **savefig_kwargs):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs.setdefault("format", "svg")
    savefig_kwargs.setdefault("bbox_inches", "tight")
    annotation_paths = collect_annotation_svgs(fig)

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        raw_path = Path(tmp.name)
    try:
        fig.savefig(raw_path, **savefig_kwargs)
        inject_swatch_patterns(raw_path, output_path=output_path, tile_width_pt=tile_width_pt)
        if annotation_paths:
            inject_annotation_previews(output_path, annotation_paths=annotation_paths, output_path=output_path)
    finally:
        raw_path.unlink(missing_ok=True)
    return output_path


def available_svg_swatches():
    swatch_dir = resources.files("pystrat").joinpath("swatches_svg")
    return sorted(int(path.name.split(".")[0]) for path in swatch_dir.iterdir() if path.name.endswith(".svg"))


def plot_svg_swatch_catalog(output_path, columns=10):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codes = available_svg_swatches()
    rows = math.ceil(len(codes) / columns)
    cell_w, cell_h = 1.0, 1.12
    box_size = 0.82
    fig_w, fig_h = columns * 0.86, rows * 0.94

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, columns * cell_w)
    ax.set_ylim(rows * cell_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for idx, code in enumerate(codes):
        row, col = divmod(idx, columns)
        x = col * cell_w + (cell_w - box_size) / 2
        y = row * cell_h + 0.08
        rect = Rectangle((x, y), box_size, box_size, facecolor="none", edgecolor="0.15", linewidth=0.45)
        rect.set_gid(make_swatch_gid(code, idx))
        ax.add_patch(rect)
        ax.text(col * cell_w + 0.5, y + box_size + 0.12, str(code), ha="center", va="top", fontsize=6.0)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        raw_path = Path(tmp.name)
    try:
        fig.savefig(raw_path, format="svg", bbox_inches="tight", pad_inches=0.04)
        inject_swatch_previews(raw_path, output_path=output_path)
    finally:
        raw_path.unlink(missing_ok=True)
        plt.close(fig)
    return output_path
