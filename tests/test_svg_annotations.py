from importlib import resources

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pystrat


def test_svg_annotation_export_is_vector(tmp_path):
    fig, ax = plt.subplots(figsize=(2, 2))
    annotation_path = resources.files("pystrat").joinpath("annotations", "stromatolite.png")

    pystrat.plot_annotation(annotation_path, [0.1, 0.1], 0.8, ax, backend="svg")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output_path = tmp_path / "annotation.svg"
    pystrat.savefig_svg(fig, output_path)
    plt.close(fig)

    svg_text = output_path.read_text()
    assert "<image" not in svg_text
    assert "pystrat-annotation-" in svg_text
    assert "ann-preview-" in svg_text