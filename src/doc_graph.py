import copy
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.utils.graph_utils import load_graph, save_graph
from src.utils.graph_patterns import (
    _PARAGRAPH_TYPES, _CAPTION_TYPES, _ELEM_TYPE_TO_CATEGORY,
    _HEADER_TYPES, _CAPTIONABLE_TYPES, _REF_PATTERNS,
    _SECTION_MEMBER_TYPES,
    )

# helpers

def _centroid(elem: dict) -> Tuple[float, float]:
    """Return (cx, cy) from the element.  Prefers the ``centroid`` key that
    ``LayoutExtractor`` already stores; falls back to computing from ``box``."""
    if "centroid" in elem and elem["centroid"]:
        c = elem["centroid"]
        return (float(c[0]), float(c[1]))
    box = elem.get("box", {})
    cx = (box.get("x1", 0) + box.get("x2", 0)) / 2.0
    cy = (box.get("y1", 0) + box.get("y2", 0)) / 2.0
    return (cx, cy)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sort_key_reading_order(elem: dict) -> Tuple[float, float]:
    """Sort key that approximates top-to-bottom, left-to-right reading order
    *within a single column*.  For multi-column pages use
    :func:`_compute_reading_order_ranks`, which lays out columns correctly."""
    box = elem.get("box", {})
    return (box.get("y1", 0), box.get("x1", 0))


def _detect_column_split(
    cxs: List[float], content_width: float, gap_ratio: float,
) -> Optional[float]:
    """Find an x split between two columns from a set of element x-centroids.

    Data-driven: looks for the largest horizontal gap between consecutive
    centroids.  If that gap is wide enough (``>= gap_ratio * content_width``)
    it is treated as the gutter and the midpoint is returned; otherwise the
    band is single-column and ``None`` is returned.
    """
    if len(cxs) < 2 or content_width <= 0:
        return None
    s = sorted(cxs)
    best_gap = 0.0
    best_mid: Optional[float] = None
    for a, b in zip(s, s[1:]):
        gap = b - a
        if gap > best_gap:
            best_gap = gap
            best_mid = (a + b) / 2.0
    if best_mid is not None and best_gap >= gap_ratio * content_width:
        return best_mid
    return None


def _order_band(
    band: List[Tuple[int, dict]], content_width: float, gap_ratio: float,
) -> List[int]:
    """Order the elements of one horizontal band (the strip between two
    full-width / spanning elements).  Detects a left/right column split and,
    when found, emits the whole left column top-to-bottom before the whole
    right column; otherwise falls back to plain reading order."""
    if not band:
        return []
    cxs = [_centroid(e)[0] for _, e in band]
    split = _detect_column_split(cxs, content_width, gap_ratio)
    if split is None:
        return [idx for idx, _ in
                sorted(band, key=lambda t: _sort_key_reading_order(t[1]))]
    left  = [(idx, e) for idx, e in band if _centroid(e)[0] <= split]
    right = [(idx, e) for idx, e in band if _centroid(e)[0] >  split]
    left.sort(key=lambda t: _sort_key_reading_order(t[1]))
    right.sort(key=lambda t: _sort_key_reading_order(t[1]))
    return [idx for idx, _ in left] + [idx for idx, _ in right]


def _order_page_elements(
    items: List[Tuple[int, dict]], *, span_ratio: float, gap_ratio: float,
) -> List[int]:
    """Return the indices of one page's elements in column-aware reading order.

    Algorithm:
      1. Classify elements as *spanning* (box width >= ``span_ratio`` of the
         page content width — section titles, full-width figures / tables) or
         *columnar* (everything else).
      2. Spanning elements act as horizontal separators: they divide the page
         into bands.  Each columnar element joins the band defined by how many
         spanning elements sit above it (by vertical centroid).
      3. Each band is ordered with :func:`_order_band` (left column, then right
         column); bands and spanning elements are emitted top-to-bottom.

    On a single-column page no left/right gutter is found, so the result is the
    usual top-to-bottom order.
    """
    if not items:
        return []

    boxes = {idx: elem.get("box", {}) for idx, elem in items}
    content_left  = min(b.get("x1", 0) for b in boxes.values())
    content_right = max(b.get("x2", 0) for b in boxes.values())
    content_width = content_right - content_left

    spanning: List[Tuple[int, dict]] = []
    columnar: List[Tuple[int, dict]] = []
    for idx, elem in items:
        b = boxes[idx]
        width = b.get("x2", 0) - b.get("x1", 0)
        if content_width > 0 and width >= span_ratio * content_width:
            spanning.append((idx, elem))
        else:
            columnar.append((idx, elem))

    spanning.sort(key=lambda t: _centroid(t[1])[1])
    span_ys = [_centroid(e)[1] for _, e in spanning]

    bands: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, elem in columnar:
        cy = _centroid(elem)[1]
        band = sum(1 for sy in span_ys if sy < cy)
        bands[band].append((idx, elem))

    result: List[int] = []
    for b in range(len(spanning) + 1):
        result.extend(_order_band(bands.get(b, []), content_width, gap_ratio))
        if b < len(spanning):
            result.append(spanning[b][0])
    return result


def _compute_reading_order_ranks(
    report: List[dict], *, span_ratio: float = 0.8, gap_ratio: float = 0.2,
) -> Dict[int, int]:
    """Compute a document-global reading-order rank for every element index.

    Pages are processed in order; within each page elements are laid out with
    :func:`_order_page_elements` so that two-column layouts read down the left
    column before the right.  The returned mapping ``idx -> rank`` is the single
    source of truth for every ordering-dependent edge builder.
    """
    page_elements: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, elem in enumerate(report):
        page_elements[elem.get("page_num", 0)].append((idx, elem))

    ranks: Dict[int, int] = {}
    rank = 0
    for pn in sorted(page_elements):
        for idx in _order_page_elements(
            page_elements[pn], span_ratio=span_ratio, gap_ratio=gap_ratio,
        ):
            ranks[idx] = rank
            rank += 1
    return ranks


def _extract_reference_number_from_caption(caption_text: str) -> Optional[int]:
    """Try to pull a number out of a caption string, e.g.
    'Figure 3: Comparison of …' → 3."""
    m = re.search(r"(\d+)", caption_text or "")
    return int(m.group(1)) if m else None


def build_document_graph(
    report: List[dict],
    *,
    header_paragraph_max_distance: float = 150.0,
    header_paragraph_same_page: bool = True,
    add_reading_order: bool = True,
    add_caption_edges: bool = True,
    add_header_paragraph_edges: bool = False,
    add_section_grouping_edges: bool = True,
    section_grouping_cross_page: bool = True,
    add_text_reference_edges: bool = True,
    add_page_sequence_edges: bool = True,
    column_span_ratio: float = 0.8,
    column_gap_ratio: float = 0.2,
) -> dict:
    """Build a graph from the segmentation report.

    Args:
        report (list[dict]): The JSON list produced by ``run_segmentation()``
            (each element is a dict with keys like ``name``, ``page_num``,
            ``box``, ``text``, ``description``, ``confidence``, ``centroid``,
            ``bbox_idx``, …).
        header_paragraph_max_distance (float): Maximum vertical pixel distance
            between a title and a paragraph for them to be linked.
        header_paragraph_same_page (bool): If True, headers and paragraphs are
            only linked when they share the same page.
        add_reading_order (bool): Add ``reading_order`` edges between
            consecutive elements per page.
        add_caption_edges (bool): Add ``caption_of`` edges linking captions to
            their figures / tables.
        add_header_paragraph_edges (bool): Add ``header_paragraph`` edges
            linking headers to nearby paragraphs by vertical distance.
            Disabled by default: it is superseded by the reading-order based
            ``section_member`` edges (see ``add_section_grouping_edges``).
        add_section_grouping_edges (bool): Add ``section_member`` edges linking
            each title to the plain-text / table / figure / formula elements
            that follow it in reading order, until the next title is reached.
        section_grouping_cross_page (bool): If True, a title's section continues
            onto following pages until the next title; if False, grouping is
            restricted to the title's own page.
        add_text_reference_edges (bool): Add ``text_references`` edges linking
            paragraphs to figures / tables mentioned via patterns like "Fig. 1".
        add_page_sequence_edges (bool): Add ``next_page`` edges between
            consecutive page nodes in document order.
        column_span_ratio (float): An element whose box width is at least this
            fraction of the page content width is treated as *spanning* (e.g. a
            section title or full-width figure) rather than belonging to a
            single column.  Used for column-aware reading order.
        column_gap_ratio (float): Minimum horizontal gap between element
            x-centroids — as a fraction of the page content width — for it to be
            recognised as the gutter between two columns.

    Returns:
        dict: ``{"nodes": [...], "edges": [...], "meta": {...}}``
    """

    nodes: Dict[str, dict] = {}      # node_id → node dict
    edges: List[dict] = []           # list of edge dicts

    # Document-global, column-aware reading order; the single source of truth
    # for every ordering-dependent edge builder below.
    order_ranks = _compute_reading_order_ranks(
        report, span_ratio=column_span_ratio, gap_ratio=column_gap_ratio,
    )

    page_nums = sorted({e.get("page_num", 0) for e in report})

    for pn in page_nums:
        nid = f"page_{pn}"
        nodes[nid] = {
            "node_id":   nid,
            "node_type": "page",
            "page_num":  pn,
            "callers":   [],
            "callees":   [],
        }

    elem_id_map: Dict[int, str] = {}      # list-index → node_id

    for idx, elem in enumerate(report):
        nid = f"elem_{elem.get('bbox_idx', idx)}"
        node = dict(elem)                  # copy all original keys
        node["node_id"]   = nid
        node["node_type"] = "element"
        node["callers"]   = []
        node["callees"]   = []
        nodes[nid] = node
        elem_id_map[idx] = nid

    for idx, elem in enumerate(report):
        pn = elem.get("page_num", 0)
        page_nid = f"page_{pn}"
        elem_nid = elem_id_map[idx]
        _add_edge(nodes, edges, page_nid, elem_nid, "page_contains")

    if add_caption_edges:
        _build_caption_edges(report, elem_id_map, nodes, edges)

    if add_header_paragraph_edges:
        _build_header_paragraph_edges(
            report, elem_id_map, nodes, edges,
            max_dist=header_paragraph_max_distance,
            same_page=header_paragraph_same_page,
        )

    if add_section_grouping_edges:
        _build_section_grouping_edges(
            report, elem_id_map, nodes, edges, order_ranks,
            cross_page=section_grouping_cross_page,
        )

    if add_text_reference_edges:
        _build_text_reference_edges(report, elem_id_map, nodes, edges, order_ranks)

    if add_reading_order:
        _build_reading_order_edges(report, elem_id_map, nodes, edges, order_ranks)

    if add_page_sequence_edges:
        _build_page_sequence_edges(page_nums, nodes, edges)

    meta = {
        "total_pages":    len(page_nums),
        "total_elements": len(report),
        "total_nodes":    len(nodes),
        "total_edges":    len(edges),
        "edge_type_counts": _count_edge_types(edges),
    }

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "meta":  meta,
    }


# edge helpers

def _add_edge(
    nodes: Dict[str, dict],
    edges: List[dict],
    source_id: str,
    target_id: str,
    edge_type: str,
    **extra,
) -> None:
    """Create an edge and update callers / callees on both nodes."""
    edge = {
        "source":    source_id,
        "target":    target_id,
        "edge_type": edge_type,
        **extra,
    }
    edges.append(edge)
    if source_id in nodes:
        nodes[source_id]["callees"].append(target_id)
    if target_id in nodes:
        nodes[target_id]["callers"].append(source_id)


def _count_edge_types(edges: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for e in edges:
        counts[e["edge_type"]] += 1
    return dict(counts)


# caption linking

def _build_caption_edges(
    report: List[dict],
    elem_id_map: Dict[int, str],
    nodes: Dict[str, dict],
    edges: List[dict],
) -> None:
    """Link caption elements to their nearest captionable element on the
    same page.  This mirrors the logic already in
    ``LayoutExtractor._detect_captions_around_figures`` but works on the
    final report so it can also catch captions that were merged."""

    page_elements: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, elem in enumerate(report):
        page_elements[elem.get("page_num", 0)].append((idx, elem))

    for pn, elems in page_elements.items():
        captions = [(i, e) for i, e in elems if e.get("name") in _CAPTION_TYPES]
        targets  = [(i, e) for i, e in elems if e.get("name") in _CAPTIONABLE_TYPES]

        for ci, cap in captions:
            cap_center = _centroid(cap)
            best_idx = None
            best_dist = float("inf")
            for ti, tgt in targets:
                d = _distance(cap_center, _centroid(tgt))
                if d < best_dist:
                    best_dist = d
                    best_idx = ti
            if best_idx is not None:
                _add_edge(
                    nodes, edges,
                    elem_id_map[ci], elem_id_map[best_idx],
                    "caption_of",
                    distance=round(best_dist, 2),
                )



def _build_header_paragraph_edges(
    report: List[dict],
    elem_id_map: Dict[int, str],
    nodes: Dict[str, dict],
    edges: List[dict],
    *,
    max_dist: float = 150.0,
    same_page: bool = True,
) -> None:
    """Link each ``title`` to the closest subsequent ``plain text`` elements
    that are within *max_dist* vertical pixels.

    The heuristic: walk downwards from each header.  Every paragraph whose
    top edge is below the header's bottom edge *and* within ``max_dist`` px
    is linked — until we hit another header or exceed the distance.
    """

    page_elements: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, elem in enumerate(report):
        page_elements[elem.get("page_num", 0)].append((idx, elem))

    for pn, elems in page_elements.items():
        sorted_elems = sorted(elems, key=lambda t: _sort_key_reading_order(t[1]))

        current_header_idx: Optional[int] = None
        current_header_bottom: float = 0.0

        for idx, elem in sorted_elems:
            etype = elem.get("name", "")

            if etype in _HEADER_TYPES:
                current_header_idx = idx
                box = elem.get("box", {})
                current_header_bottom = box.get("y2", 0)

            elif etype in _PARAGRAPH_TYPES and current_header_idx is not None:
                box = elem.get("box", {})
                para_top = box.get("y1", 0)
                vdist = para_top - current_header_bottom

                if 0 <= vdist <= max_dist:
                    _add_edge(
                        nodes, edges,
                        elem_id_map[current_header_idx],
                        elem_id_map[idx],
                        "header_paragraph",
                        vertical_distance=round(vdist, 2),
                    )
                elif vdist > max_dist:
                    current_header_idx = None


# section grouping (reading-order based)

def _build_section_grouping_edges(
    report: List[dict],
    elem_id_map: Dict[int, str],
    nodes: Dict[str, dict],
    edges: List[dict],
    order_ranks: Dict[int, int],
    *,
    cross_page: bool = True,
) -> None:
    """Group content elements under the title that heads their section.

    Heuristic (reading-order based, not distance based): walk every element in
    document reading order.  Each time a ``title`` is seen it becomes the
    *current section head*.  Every subsequent content element (plain text,
    table, figure, formula and their captions) is linked to that title with a
    ``section_member`` edge — because they form one uninterrupted reading-order
    sequence after the title — until the next title is reached, which starts a
    new section.

    This generalises the old distance-based ``header_paragraph`` link: tables
    and figures that sit far below their title are still grouped correctly,
    while encountering the next title cleanly terminates the previous section.

    Args:
        cross_page: If True, a section continues onto following pages until the
            next title appears.  If False, only elements on the title's own page
            are grouped (content on a later page with no title of its own is
            left ungrouped).
    """

    # document reading order: pages in order, column-aware reading order within
    # each page (see _compute_reading_order_ranks).
    ordered = sorted(range(len(report)), key=lambda i: order_ranks[i])

    current_title_idx: Optional[int] = None
    current_title_page: Optional[int] = None

    for idx in ordered:
        elem = report[idx]
        etype = elem.get("name", "")

        if etype in _HEADER_TYPES:
            current_title_idx = idx
            current_title_page = elem.get("page_num", 0)
            continue

        if current_title_idx is None:
            continue
        if etype not in _SECTION_MEMBER_TYPES:
            continue
        if not cross_page and elem.get("page_num", 0) != current_title_page:
            continue

        _add_edge(
            nodes, edges,
            elem_id_map[current_title_idx],
            elem_id_map[idx],
            "section_member",
            member_type=etype,
        )


# text-reference edges (regex)

def _build_text_reference_edges(
    report: List[dict],
    elem_id_map: Dict[int, str],
    nodes: Dict[str, dict],
    edges: List[dict],
    order_ranks: Dict[int, int],
) -> None:
    """Scan every ``plain text`` element's ``text`` field for patterns like
    "Fig. 1", "Table 2", etc.  When a match is found, link the paragraph
    to the *N*-th figure / table / formula in the document (counting in
    reading order).

    Numbering is 1-based and document-global (e.g., "Figure 3" is the 3rd
    figure encountered across all pages).  When a caption contains a number
    we prefer that; otherwise we fall back to positional counting.
    """

    # --- build lookup: category → ordinal → element index ----------------
    category_by_number:   Dict[str, Dict[int, int]] = defaultdict(dict)
    category_positional:  Dict[str, List[int]]       = defaultdict(list)

    # collect caption texts keyed by captionable element index
    caption_texts: Dict[int, str] = {}

    page_elems: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, elem in enumerate(report):
        page_elems[elem.get("page_num", 0)].append((idx, elem))

    for pn, elems in page_elems.items():
        captions = [(i, e) for i, e in elems if e.get("name") in _CAPTION_TYPES]
        targets  = [(i, e) for i, e in elems if e.get("name") in _CAPTIONABLE_TYPES]
        for ci, cap in captions:
            cc = _centroid(cap)
            best_i = None
            best_d = float("inf")
            for ti, tgt in targets:
                d = _distance(cc, _centroid(tgt))
                if d < best_d:
                    best_d = d
                    best_i = ti
            if best_i is not None:
                cap_text = cap.get("text", "") or cap.get("description", "")
                caption_texts[best_i] = cap_text

    # order captionable elements and assign numbers
    all_captionables = [
        (idx, elem) for idx, elem in enumerate(report)
        if elem.get("name") in _CAPTIONABLE_TYPES
    ]
    all_captionables.sort(key=lambda t: order_ranks[t[0]])

    positional_counter: Dict[str, int] = defaultdict(int)
    for idx, elem in all_captionables:
        cat = _ELEM_TYPE_TO_CATEGORY.get(elem.get("name", ""), "")
        if not cat:
            continue
        positional_counter[cat] += 1
        category_positional[cat].append(idx)

        cap_text = caption_texts.get(idx, "")
        num = _extract_reference_number_from_caption(cap_text)
        if num is not None:
            category_by_number[cat][num] = idx

    already_linked: set = set()

    for idx, elem in enumerate(report):
        if elem.get("name") not in _PARAGRAPH_TYPES:
            continue
        text = (elem.get("text", "") or "") + " " + (elem.get("description", "") or "")
        if not text.strip():
            continue

        for ref_cat, pattern in _REF_PATTERNS:
            for match in pattern.finditer(text):
                ref_num = int(match.group(1))
                target_idx: Optional[int] = None
                if ref_num in category_by_number.get(ref_cat, {}):
                    target_idx = category_by_number[ref_cat][ref_num]
                elif 1 <= ref_num <= len(category_positional.get(ref_cat, [])):
                    target_idx = category_positional[ref_cat][ref_num - 1]

                if target_idx is not None:
                    pair = (idx, target_idx)
                    if pair not in already_linked:
                        already_linked.add(pair)
                        _add_edge(
                            nodes, edges,
                            elem_id_map[idx],
                            elem_id_map[target_idx],
                            "text_references",
                            matched_pattern=match.group(0),
                            reference_number=ref_num,
                            reference_category=ref_cat,
                        )


def _build_reading_order_edges(
    report: List[dict],
    elem_id_map: Dict[int, str],
    nodes: Dict[str, dict],
    edges: List[dict],
    order_ranks: Dict[int, int],
) -> None:
    """Chain elements on each page in column-aware reading order (down the left
    column, then the right; see _compute_reading_order_ranks)."""

    page_elements: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, elem in enumerate(report):
        page_elements[elem.get("page_num", 0)].append((idx, elem))

    for pn in sorted(page_elements):
        elems = sorted(page_elements[pn], key=lambda t: order_ranks[t[0]])
        for i in range(len(elems) - 1):
            _add_edge(
                nodes, edges,
                elem_id_map[elems[i][0]],
                elem_id_map[elems[i + 1][0]],
                "reading_order",
            )


def _build_page_sequence_edges(
    page_nums: List[int],
    nodes: Dict[str, dict],
    edges: List[dict],
) -> None:
    """Chain page nodes in document order with ``next_page`` edges."""
    for i in range(len(page_nums) - 1):
        _add_edge(
            nodes, edges,
            f"page_{page_nums[i]}",
            f"page_{page_nums[i + 1]}",
            "next_page",
        )


def run_graph_building(
    report_path: str = None,
    output_path: Optional[str] = None,
    **seg_kwargs,
) -> Tuple[List[dict], dict]:
    """Convenience wrapper: runs segmentation then builds the graph.

    Args:
        report_path (str): path to JSON obtained with segmentation
        output_path (str or None): Where to save the graph JSON.
            If None, defaults to ``output_path`` with ``_graph`` suffix.
        **seg_kwargs: Forwarded to ``run_segmentation()`` (e.g. ``device``,
            ``use_api``, ``use_async``, ``max_concurrent``).

    Returns:
        tuple: (report, graph)
    """

    if report_path is None:
        raise NotImplementedError
    else:
        report = load_graph(report_path)

    graph = build_document_graph(report)

    save_graph(graph, output_path)
    print(f"Graph saved to {output_path}  "
          f"({graph['meta']['total_nodes']} nodes, "
          f"{graph['meta']['total_edges']} edges)")

    return report, graph

