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
    _HEADER_TYPES, _CAPTIONABLE_TYPES, _REF_PATTERNS
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
    """Sort key that approximates top-to-bottom, left-to-right reading order."""
    box = elem.get("box", {})
    return (box.get("y1", 0), box.get("x1", 0))


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
    add_header_paragraph_edges: bool = True,
    add_text_reference_edges: bool = True,
    add_page_sequence_edges: bool = True,
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
            linking headers to nearby paragraphs.
        add_text_reference_edges (bool): Add ``text_references`` edges linking
            paragraphs to figures / tables mentioned via patterns like "Fig. 1".
        add_page_sequence_edges (bool): Add ``next_page`` edges between
            consecutive page nodes in document order.

    Returns:
        dict: ``{"nodes": [...], "edges": [...], "meta": {...}}``
    """

    nodes: Dict[str, dict] = {}      # node_id → node dict
    edges: List[dict] = []           # list of edge dicts

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

    if add_text_reference_edges:
        _build_text_reference_edges(report, elem_id_map, nodes, edges)

    if add_reading_order:
        _build_reading_order_edges(report, elem_id_map, nodes, edges)

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


# text-reference edges (regex)

def _build_text_reference_edges(
    report: List[dict],
    elem_id_map: Dict[int, str],
    nodes: Dict[str, dict],
    edges: List[dict],
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
    all_captionables.sort(key=lambda t: (t[1].get("page_num", 0),
                                         _sort_key_reading_order(t[1])))

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
) -> None:
    """Chain elements on each page in reading order (top → bottom,
    left → right)."""

    page_elements: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, elem in enumerate(report):
        page_elements[elem.get("page_num", 0)].append((idx, elem))

    for pn in sorted(page_elements):
        elems = sorted(page_elements[pn],
                       key=lambda t: _sort_key_reading_order(t[1]))
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


# run_graph_building("/home/lameus/Projects/layout-aware-doc-segmentation/data/kerr_part.json",
#                    "/home/lameus/Projects/layout-aware-doc-segmentation/data/kerr_part_graph_1.json")