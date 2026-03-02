import os
import math
import cv2
import json
import base64

from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
import numpy as np
from tqdm import tqdm

import torch
from utils import utils


class LayoutExtractor:
    def __init__(
        self,
        device: str = "cuda",
        save_img: bool = False,
        model_name: str = "juliozhao/DocLayout-YOLO-DocStructBench",
        weights_name: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
    ) -> None:
        """Class for bbox extraction and document parts processing

        Args:
            device (str, optional): device type "cpu"/"cuda". Defaults to "cuda".
            save_img (bool, optional): an argument to define save processed images or not. Defaults to False.
            model_name (str, optional): model name. Defaults to "juliozhao/DocLayout-YOLO-DocStructBench".
            weights_name (str, optional): weights name. Defaults to "doclayout_yolo_docstructbench_imgsz1024.pt".
        """
        # Load the pre-trained model
        model_path = hf_hub_download(repo_id=model_name, filename=weights_name)
        self.device = device
        self.save_img = save_img

        self.model = YOLOv10(model_path).to(device)

        self.labels = {
            0: "title",
            1: "plain text",
            2: "abandon",
            3: "figure",
            4: "figure_caption",
            5: "table",
            6: "table_caption",
            7: "table_footnote",
            8: "isolate_formula",
            9: "formula_caption",
        }

        self.label_mapping = {
            "figure_caption": "figure",
            "table_caption": "table",
            "table_footnote": "table",
            "formula_caption": "isolate_formula",
        }

        self.doc_path = None

        # Parts concatenation related
        self.captions = []
        self.specific_elems = []
        self.relevant_pairs = []

    def get_bboxes(
        self,
        doc_path: str,
        output_path: str = None,
        # img_size: int = 1024,
        conf_threshold: float = 0.15,
    ) -> list:
        """The main method to call the YOLO model and extract bboxes.
        It provides self.pages, which collects arrays for every page
        as an image in the case of a PDF document. It  returns bboxes
        represented in JSON-like format.

        Args:
            doc_path (str): path to the document to process, it can be image, or pdf.
            output_path (str, optional): path to the file to save returned result. Defaults to None.
            conf_threshold (float, optional): minimum confidence level of YOLO model. Defaults to 0.15.

        Returns:
            bbox_json: [{'name': str,
                        'class': int,
                        'confidence': float,
                        'box': {'x1': float,
                        'y1': float,
                        'x2': float,
                        'y2': float}]
        """
        self.doc_path = doc_path

        self.pages = []
        self.bboxes = []

        if ".pdf" in self.doc_path:
            self.pages = utils.process_pdf(self.doc_path)
            self.bboxes = self.model.predict(
                self.pages, conf=conf_threshold, device=self.device
            )
        else:
            bbox_result = self.model.predict(
                doc_path, conf=conf_threshold, device=self.device
            )
            self.bboxes = (
                bbox_result if isinstance(bbox_result, list) else [bbox_result]
            )

        return self._process_bbox_results(output_path)

    def get_bboxes_batched(
        self,
        doc_path: str,
        output_path: str = None,
        conf_threshold: float = 0.15,
        batch_size: int = 5,
        max_pages: int = None,
    ) -> list:
        """Process large documents in batches to manage CUDA memory"""

        self.doc_path = doc_path
        self.pages = []
        self.bboxes = []

        if ".pdf" in self.doc_path:
            self.pages = utils.process_pdf(self.doc_path)

            if max_pages:
                self.pages = self.pages[:max_pages]

            for i in tqdm(range(0, len(self.pages), batch_size)):
                batch_pages = self.pages[i : i + batch_size]

                batch_bboxes = self.model.predict(
                    batch_pages, conf=conf_threshold, device=self.device, verbose=False
                )
                self.bboxes.extend(batch_bboxes)

                if self.device == "cuda":
                    torch.cuda.empty_cache()
        else:
            # Single image processing
            bbox_result = self.model.predict(
                doc_path, conf=conf_threshold, device=self.device
            )
            self.bboxes = (
                bbox_result if isinstance(bbox_result, list) else [bbox_result]
            )

        return self._process_bbox_results(output_path)

    def _process_bbox_results(self, output_path: str = None) -> list:
        """Process bbox results after prediction (shared logic for both methods)"""

        if self.save_img:
            # Annotate and save the result
            for i in range(len(self.bboxes)):
                annotated_frame = self.bboxes[i].plot(
                    pil=True, line_width=5, font_size=20
                )
                annot_name = os.path.basename(self.doc_path).split(".")
                annot_name = annot_name[0] + "_layout" + "_" + str(i) + ".jpg"
                if output_path:
                    cv2.imwrite(
                        os.path.join(os.path.dirname(output_path), annot_name),
                        annotated_frame,
                    )

        self.bbox_json = []
        for i in tqdm(range(len(self.bboxes))):
            current_bbox_json = json.loads(self.bboxes[i].tojson())
            current_bbox_json = self.merge_duplicated(
                bbox_json=current_bbox_json, iou_threshold=0.75
            )
            for j in range(len(current_bbox_json)):
                current_bbox_json[j]["page_num"] = i

            self.bbox_json.extend(current_bbox_json)

        for i in range(len(self.bbox_json)):
            self.bbox_json[i]["centroid"] = self._get_centoid(bbox_idx=i)
            # TODO: it can be executed after parts concatenation
            self.bbox_json[i]["image_bytes"] = self.encode_image(bbox_idx=i)
            self.bbox_json[i]["bbox_idx"] = i

            # specific symbols from the pages, not supposed to be analyzed
            if self.bbox_json[i]["name"] == "abandon":
                self.bbox_json[i]["ignore"] = True
            else:
                self.bbox_json[i]["ignore"] = False

        if output_path:
            for i in range(len(self.bbox_json)):
                self.bbox_json[i]["image_bytes"] = str(self.bbox_json[i]["image_bytes"])

            with open(output_path + ".json", "w") as f:
                json.dump(self.bbox_json, f)

        return self.bbox_json

    # TODO: drop it to utils to avoid rewriting it in doc_ocr
    def encode_image(self, bbox_idx):
        _bbox = self.bbox_json[bbox_idx]["box"]
        x1, y1, x2, y2 = [i for i in _bbox.values()]

        x = math.floor(x1)
        y = math.floor(y1)
        w = math.ceil(x2) - x
        h = math.ceil(y2) - y

        if ".pdf" in self.doc_path:
            image = self.pages[self.bbox_json[bbox_idx]["page_num"]]

        else:
            image = cv2.imread(self.doc_path)

        cropped_img = image[y : y + h, x : x + w]
        retval, buffer = cv2.imencode(".jpg", cropped_img)
        jpg_as_bytes = base64.b64encode(buffer)
        return jpg_as_bytes

    def merge_duplicated(self, bbox_json, iou_threshold: float = 0.75) -> list:
        """Method to merge overlaped bboxes. It considers the Intersection over Union (iou)
        and save a bbox with the higher confidence level.

        Args:
            bbox_json (list): result of get_bboxes
            iou_threshold (float, optional): Intersection over Union level. Defaults to 0.75.

        Returns:
            bbox_json: [{'name': str,
                        'class': int,
                        'confidence': float,
                        'box': {'x1': float,
                        'y1': float,
                        'x2': float,
                        'y2': float}]
        """
        name_groups = {}
        for box in bbox_json:
            name = box["name"]
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(box)

        merged_boxes = []

        for name, group in name_groups.items():
            group.sort(key=lambda x: -x["confidence"])
            i = 0
            while i < len(group):
                current = group[i]
                j = i + 1
                while j < len(group):
                    candidate = group[j]
                    iou = self._calculate_iou(current["box"], candidate["box"])
                    if iou > iou_threshold:
                        current = self._merge_two_boxes(current, candidate)
                        group.pop(j)
                    else:
                        j += 1
                merged_boxes.append(current)
                i += 1

        bbox_json = merged_boxes
        return bbox_json

    def _calculate_iou(self, bbox1, bbox2):
        # Determine coordinates of intersection rectangle
        x_left = max(bbox1["x1"], bbox2["x1"])
        y_top = max(bbox1["y1"], bbox2["y1"])
        x_right = min(bbox1["x2"], bbox2["x2"])
        y_bottom = min(bbox1["y2"], bbox2["y2"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bbox1_area = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
        bbox2_area = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])

        union_area = bbox1_area + bbox2_area - intersection_area

        iou = intersection_area / union_area
        return iou

    def _merge_two_boxes(self, bbox1, bbox2) -> list:
        # Keep box with higher confidence
        merged = bbox1 if bbox1["confidence"] >= bbox2["confidence"] else bbox2

        merged_box_coords = {
            "x1": min(bbox1["box"]["x1"], bbox2["box"]["x1"]),
            "y1": min(bbox1["box"]["y1"], bbox2["box"]["y1"]),
            "x2": max(bbox1["box"]["x2"], bbox2["box"]["x2"]),
            "y2": max(bbox1["box"]["y2"], bbox2["box"]["y2"]),
        }

        merged["confidence"] = max(bbox1["confidence"], bbox2["confidence"])
        merged["box"] = merged_box_coords

        return merged

    def _get_centoid(self, bbox_idx):
        bbox_coors = self.bbox_json[bbox_idx]["box"]
        return (
            (bbox_coors["x1"] + bbox_coors["x2"]) / 2,
            (bbox_coors["y1"] + bbox_coors["y2"]) / 2,
        )

    def _find_closest_bboxes(self):
        """Find related bboxes based on label_mapping and cosine similarity
        of centroids.
        """
        for i in range(len(self.bbox_json)):
            if self.bbox_json[i]["name"] in self.label_mapping:
                self.captions.append(self.bbox_json[i])
                self.bbox_json[i]["ignore"] = True  # because it'll be concatenated
            elif self.bbox_json[i]["name"] in self.label_mapping.values():
                self.specific_elems.append(self.bbox_json[i])

        for caption_id, caption in enumerate(self.captions):
            max_similarity = -1
            closest_elem_id = None

            for elem_id, elem in enumerate(self.specific_elems):
                if elem["name"] != self.label_mapping[caption["name"]]:
                    continue

                if elem["page_num"] != caption["page_num"]:
                    continue

                similarity = utils.cosine_similarity(
                    caption["centroid"], elem["centroid"]
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_elem_id = elem_id

            if closest_elem_id is not None:
                self.relevant_pairs.append(
                    {
                        "caption_id": caption_id,
                        "element_id": closest_elem_id,
                        "similarity": max_similarity,
                    }
                )

    def _merge_related_bboxes(self):
        """Method to merge related pairs of bboxes, which are found with _find_closest_bboxes."""
        for pair in self.relevant_pairs:
            caption_id = pair["caption_id"]
            element_id = pair["element_id"]

            try:
                part1 = utils.readb64(eval(self.captions[caption_id]["image_bytes"]))
            except SyntaxError:
                part1 = utils.readb64(self.captions[caption_id]["image_bytes"])
            try:
                part2 = utils.readb64(
                    eval(self.specific_elems[element_id]["image_bytes"])
                )
            except SyntaxError:
                part2 = utils.readb64(self.specific_elems[element_id]["image_bytes"])

            height1, width1 = part1.shape[:2]
            height2, width2 = part2.shape[:2]

            if width1 != width2:
                if width1 < width2:
                    new_height = int(height1 * (width2 / width1))
                    part1 = cv2.resize(part1, (width2, new_height))
                else:
                    new_height = int(height2 * (width1 / width2))
                    part2 = cv2.resize(part2, (width1, new_height))

            merged_parts = cv2.vconcat([part1, part2])

            # rewrite the specific element with merged
            changed_id = self.specific_elems[element_id]["bbox_idx"]
            self.bbox_json[changed_id]["image_bytes"] = utils.encode_bbox(merged_parts)

    def _merge_adjacent_plain_texts(
        self, bboxes: list = None, distance_ratio: float = 0.18, abs_gap_px: int = 18
    ):
        """ """

        if bboxes is None:
            bboxes = self.bbox_json

        def overlap_ratio_x(b1, b2):
            x_left = max(b1["x1"], b2["x1"])
            x_right = min(b1["x2"], b2["x2"])
            if x_right <= x_left:
                return 0.0
            inter = x_right - x_left
            w_min = min(max(1.0, b1["x2"] - b1["x1"]), max(1.0, b2["x2"] - b2["x1"]))
            return inter / (w_min + 1e-9)

        candidates = [
            (i, bb)
            for i, bb in enumerate(self.bbox_json)
            if (not bb.get("ignore", False))
            and bb.get("name") in ("plain text", "title")
        ]

        if not candidates:
            return

        candidates.sort(
            key=lambda t: (
                t[1].get("page_num", 0),
                t[1]["box"]["y1"],
                t[1]["box"]["x1"],
            )
        )
        idxs = [c[0] for c in candidates]
        removed = set()
        i = 0
        n = len(idxs)

        def decode_img(entry):
            raw = entry.get("image_bytes", None)
            if raw is None:
                return None
            try:
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    return utils.readb64(raw)
            except Exception:
                pass
            if isinstance(raw, str):
                try:
                    if (raw.startswith("b'") and raw.endswith("'")) or (
                        raw.startswith('b"') and raw.endswith('"')
                    ):
                        evaluated = eval(raw)
                        return utils.readb64(evaluated)
                except Exception:
                    pass
                try:
                    data = base64.b64decode(raw)
                    arr = np.frombuffer(data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    return img
                except Exception:
                    pass
            try:
                return utils.readb64(raw)
            except Exception:
                return None

        while i < n:
            idx_i = idxs[i]
            if idx_i in removed:
                i += 1
                continue

            primary_box = dict(self.bbox_json[idx_i]["box"])
            page = self.bbox_json[idx_i].get("page_num", 0)
            cluster = [idx_i]

            j = i + 1
            while j < n:
                idx_j = idxs[j]
                if idx_j in removed:
                    j += 1
                    continue
                if self.bbox_json[idx_j].get("page_num", 0) != page:
                    break

                box_j = self.bbox_json[idx_j]["box"]
                if box_j["y1"] >= primary_box["y2"]:
                    vgap = box_j["y1"] - primary_box["y2"]
                elif primary_box["y1"] >= box_j["y2"]:
                    vgap = primary_box["y1"] - box_j["y2"]
                else:
                    vgap = -min(
                        primary_box["y2"] - box_j["y1"], box_j["y2"] - primary_box["y1"]
                    )

                min_h = min(
                    max(1.0, primary_box["y2"] - primary_box["y1"]),
                    max(1.0, box_j["y2"] - box_j["y1"]),
                )
                hx = overlap_ratio_x(primary_box, box_j)

                cond_gap = vgap <= max(min_h * distance_ratio, abs_gap_px)
                cond_hx = hx >= 0.35
                cond_vert_overlap = vgap < 0

                if cond_gap and (cond_hx or cond_vert_overlap):
                    primary_box = {
                        "x1": min(primary_box["x1"], box_j["x1"]),
                        "y1": min(primary_box["y1"], box_j["y1"]),
                        "x2": max(primary_box["x2"], box_j["x2"]),
                        "y2": max(primary_box["y2"], box_j["y2"]),
                    }
                    cluster.append(idx_j)
                    removed.add(idx_j)
                    j += 1
                else:
                    break

            if len(cluster) > 1:
                primary_idx = cluster[0]
                merged_coords = primary_box

                mw = int(max(1, merged_coords["x2"] - merged_coords["x1"]))
                mh = int(max(1, merged_coords["y2"] - merged_coords["y1"]))
                canvas = np.ones((mh, mw, 3), dtype=np.uint8) * 255
                any_pasted = False

                for cid in cluster:
                    entry = self.bbox_json[cid]
                    if entry.get("name") not in ("plain text", "title"):
                        continue
                    box = entry["box"]
                    dx = int(box["x1"] - merged_coords["x1"])
                    dy = int(box["y1"] - merged_coords["y1"])
                    img = decode_img(entry)
                    if img is None:
                        continue
                    ih, iw = img.shape[:2]
                    paste_w = min(iw, mw - dx)
                    paste_h = min(ih, mh - dy)
                    if paste_w > 0 and paste_h > 0:
                        canvas[dy : dy + paste_h, dx : dx + paste_w] = img[
                            0:paste_h, 0:paste_w
                        ]
                        any_pasted = True

                if any_pasted:
                    try:
                        self.bbox_json[primary_idx]["image_bytes"] = utils.encode_bbox(
                            canvas
                        )
                    except Exception:
                        pass

                self.bbox_json[primary_idx]["box"] = merged_coords
                self.bbox_json[primary_idx]["centroid"] = (
                    (merged_coords["x1"] + merged_coords["x2"]) / 2.0,
                    (merged_coords["y1"] + merged_coords["y2"]) / 2.0,
                )

            i = j

        if not removed:
            return

        self.bbox_json = [
            entry for k, entry in enumerate(self.bbox_json) if k not in removed
        ]
        print(
            f"_merge_adjacent_plain_texts: removed {len(removed)}, total {len(self.bbox_json)}"
        )

    def _detect_captions_around_figures(
        self, bboxes: list = None, distance_ratio: float = 0.1
    ):
        """ """
        if bboxes is None:
            bboxes = self.bbox_json

        caption_for = {
            "figure": "figure_caption",
            "table": "table_caption",
            "isolate_formula": "formula_caption",
        }
        elements = [
            (i, b)
            for i, b in enumerate(bboxes)
            if (not b.get("ignore", False)) and b["name"] in caption_for.keys()
        ]
        caption_candidates = [
            (i, b)
            for i, b in enumerate(bboxes)
            if (not b.get("ignore", False))
            and b["name"]
            in (
                "plain text",
                "title",
                "figure_caption",
                "table_caption",
                "formula_caption",
            )
        ]

        def try_get_page_image(page_num):
            page_img = None
            if (
                hasattr(self, "page_images")
                and isinstance(self.page_images, dict)
                and page_num in self.page_images
            ):
                page_img = self.page_images[page_num]
            elif hasattr(self, "page_images") and isinstance(
                self.page_images, (list, tuple)
            ):
                try:
                    page_img = self.page_images[page_num]
                except Exception:
                    page_img = None
            elif hasattr(self, "page_image"):
                page_img = getattr(self, "page_image")
                if isinstance(page_img, (list, tuple)) and len(page_img) > page_num:
                    page_img = page_img[page_num]
            elif (
                hasattr(self, "page_rasters")
                and isinstance(self.page_rasters, dict)
                and page_num in self.page_rasters
            ):
                page_img = self.page_rasters[page_num]

            if page_img is not None:
                try:
                    if isinstance(page_img, (str, bytes, bytearray)):
                        try:
                            pg = (
                                utils.readb64(eval(page_img))
                                if isinstance(page_img, str)
                                and (
                                    page_img.startswith("b'")
                                    or page_img.startswith('b"')
                                )
                                else utils.readb64(page_img)
                            )
                        except Exception:
                            pg = utils.readb64(page_img)
                        page_img = pg
                except Exception:
                    pass
            return page_img

        for elem_idx, elem in elements:
            page = elem.get("page_num", 0)
            elem_box = elem["box"]
            w = elem_box["x2"] - elem_box["x1"]
            h = elem_box["y2"] - elem_box["y1"]
            expand = max(w, h) * distance_ratio
            exp = {
                "x1": elem_box["x1"] - expand,
                "y1": elem_box["y1"] - expand,
                "x2": elem_box["x2"] + expand,
                "y2": elem_box["y2"] + expand,
            }

            nearby = []
            for cand_idx, cand in caption_candidates:
                if cand_idx == elem_idx or cand.get("page_num", 0) != page:
                    continue
                c = cand["box"]
                if not (
                    c["x2"] < exp["x1"]
                    or c["x1"] > exp["x2"]
                    or c["y2"] < exp["y1"]
                    or c["y1"] > exp["y2"]
                ):
                    nearby.append((cand_idx, cand))

            if not nearby:
                continue

            all_x1 = [elem_box["x1"]] + [c[1]["box"]["x1"] for c in nearby]
            all_y1 = [elem_box["y1"]] + [c[1]["box"]["y1"] for c in nearby]
            all_x2 = [elem_box["x2"]] + [c[1]["box"]["x2"] for c in nearby]
            all_y2 = [elem_box["y2"]] + [c[1]["box"]["y2"] for c in nearby]
            merged_coords = {
                "x1": min(all_x1),
                "y1": min(all_y1),
                "x2": max(all_x2),
                "y2": max(all_y2),
            }

            page_img = try_get_page_image(page)
            if page_img is not None and isinstance(page_img, (np.ndarray,)):
                h_img, w_img = page_img.shape[:2]
                mx1 = int(max(0, merged_coords["x1"]))
                my1 = int(max(0, merged_coords["y1"]))
                mx2 = int(min(w_img, merged_coords["x2"]))
                my2 = int(min(h_img, merged_coords["y2"]))
                if mx2 > mx1 and my2 > my1:
                    crop = page_img[my1:my2, mx1:mx2].copy()
                    try:
                        self.bbox_json[elem_idx]["image_bytes"] = utils.encode_bbox(
                            crop
                        )
                    except Exception:
                        pass

            self.bbox_json[elem_idx]["box"] = merged_coords
            self.bbox_json[elem_idx]["centroid"] = (
                (merged_coords["x1"] + merged_coords["x2"]) / 2,
                (merged_coords["y1"] + merged_coords["y2"]) / 2,
            )

            for cand_idx, _ in nearby:
                if cand_idx != elem_idx:
                    # rename if needed
                    if self.bbox_json[cand_idx]["name"] in ("plain text", "title"):
                        self.bbox_json[cand_idx]["name"] = caption_for.get(
                            elem["name"], "figure_caption"
                        )
                    self.bbox_json[cand_idx]["ignore"] = True
