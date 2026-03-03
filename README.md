# layout-aware-doc-segmentation

Code and resources for the article “[Enhancing Technical Question Answering Quality through Multimodal Document Segmentation](https://ieeexplore.ieee.org/document/11359197)”.


# Document Segmentation Module

This module provides functionality for analyzing the structure of documents (images and PDFs) using computer vision and natural language processing. The main classes are `LayoutExtractor` (structure analysis) and `ImageDescription` (document element description).

## Main Components

### 1. LayoutExtractor (`doc_layout.py`)

A class for extracting structural elements from documents and processing them.

#### Core Functionality:
- Detecting bounding boxes of document elements using YOLOv10
- Processing PDFs and images
- Merging duplicate and overlapping bounding boxes
- Linking related elements (e.g., images with captions)
- Encoding images in base64

#### Key Methods:
- `get_bboxes()` — main method for retrieving bounding boxes
- `merge_duplicated()` — merges duplicate bounding boxes
- `_find_closest_bboxes()` — finds related elements (e.g., image–caption pairs)
- `_merge_related_bboxes()` — merges related elements

#### Supported Element Types:
- Titles, body text
- Images, tables, formulas
- Captions for images/tables/formulas
- Table footnotes

### 2. ImageDescription (`doc_ocr.py`)

A class for describing document elements using language models.

#### Core Functionality:
- Generating text descriptions of document elements
- Recognizing text within bounding boxes
- Supporting both API mode (via OpenAI) and local models (transformers)
- Handling both individual bounding boxes and arbitrary regions

#### Key Methods:
- `inference()` — main method for obtaining an element description
- `_parse_json()` — post-processes model output

## Usage Example

See `run.py` and `  /exasmple/segmentation_example.ipynb`
