import asyncio
import base64
import json
import time

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from src.utils import utils
from src.doc_layout import LayoutExtractor
from src.doc_ocr import ImageDescription


async def run_segmentation_batch_async(
    doc_path: str,
    output_path: str = "./book.json",
    save_img: bool = False,
    device: str = "cuda",
    use_api: bool = True,
    num_attempt: int = 10,
    max_concurrent: int = 10,
    use_context: bool = True,
):
    print("Extracting layout...")
    layout_extractor = LayoutExtractor(save_img=save_img, device=device)
    layout_extractor.get_bboxes_batched(doc_path=doc_path, output_path=output_path)
    pages = layout_extractor.pages

    layout_extractor._find_closest_bboxes()
    layout_extractor._merge_related_bboxes()
    layout_extractor._merge_adjacent_plain_texts(distance_ratio=0.25, abs_gap_px=18)
    layout_extractor._detect_captions_around_figures(distance_ratio=0.1)

    ocr_vlm = ImageDescription(
        use_api=use_api,
        use_async=True,
        pages=pages,
        bbox_json=layout_extractor.bbox_json,
        use_context=use_context,
    )

    valid_indices = [
        i for i, bbox in enumerate(layout_extractor.bbox_json)
        if not bbox.get("ignore", False)
    ]
    results_map: dict[int, str] = {}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_with_semaphore(bbox_idx: int):
        async with semaphore:
            try:
                result = await ocr_vlm.inference_async(bbox_idx=bbox_idx)
                return {"success": True, "result": result, "bbox_idx": bbox_idx}
            except Exception as e:
                return {"success": False, "error": str(e), "bbox_idx": bbox_idx}

    tasks = [_process_with_semaphore(idx) for idx in valid_indices]

    for coro in async_tqdm.as_completed(tasks, total=len(tasks)):
        res = await coro
        idx = res["bbox_idx"]
        if res["success"]:
            results_map[idx] = res["result"]
        else:
            print(f"bbox {idx} failed: {res.get('error')}")

    resulted_data = []
    for idx in valid_indices:
        if idx not in results_map:
            print(f"Permanently failed bbox {idx}")
            continue
        parsed = ocr_vlm._parse_json(results_map[idx])
        elem = layout_extractor.bbox_json[idx].copy()
        elem["description"] = parsed.get("description", "")
        elem["text"] = parsed.get("text", "")
        if isinstance(elem.get("image_bytes"), bytes):
            elem["image_bytes"] = base64.b64encode(elem["image_bytes"]).decode("utf-8")
        resulted_data.append(elem)

    resulted_data.sort(key=lambda x: x["bbox_idx"])

    utils.save_json(data=resulted_data, output_path=output_path)

    print(f"Successfully processed {len(resulted_data)}/{len(valid_indices)} bboxes")
    return resulted_data


async def run_segmentation_async(
    doc_path: str,
    output_path: str = "./book.json",
    save_img: bool = False,
    device: str = "cuda",
    use_api: bool = True,
    num_attempt: int = 10,
    max_concurrent: int = 3,
    use_async: bool = True,
    use_context: bool = True
):
    """Async version with batch processing"""
    print("Extracting layout...")
    layout_extractor = LayoutExtractor(save_img=save_img, device=device)
    
    bboxes = layout_extractor.get_bboxes_batched(
        doc_path=doc_path,
        output_path=output_path
    )
    pages = layout_extractor.pages

    layout_extractor._find_closest_bboxes()
    layout_extractor._merge_related_bboxes()
    layout_extractor._merge_adjacent_plain_texts(distance_ratio=0.25, abs_gap_px=18)
    layout_extractor._detect_captions_around_figures(distance_ratio=0.1)

    # Initialize VLM with async support
    ocr_vlm = ImageDescription(
        use_api=use_api,
        use_async=use_async,
        pages=pages,
        bbox_json=layout_extractor.bbox_json,
        use_context=use_context
    )

    valid_indices = [
        i for i, bbox in enumerate(layout_extractor.bbox_json)
        if not bbox.get("ignore", False)
    ]

    print(f"VLM processing {len(valid_indices)} bboxes...")
    
    async def process_bbox_with_retry(bbox_idx: int) -> dict:
        for attempt in range(num_attempt):
            try:
                vlm_answer = await ocr_vlm.inference_async(bbox_idx=bbox_idx)
                parsed_answer = ocr_vlm._parse_json(vlm_answer)

                elem = layout_extractor.bbox_json[bbox_idx].copy()
                elem["description"] = parsed_answer.get("description", "")
                elem["text"] = parsed_answer.get("text", "")

                if isinstance(elem["image_bytes"], bytes):
                    elem["image_bytes"] = base64.b64encode(elem["image_bytes"]).decode("utf-8")

                return {"success": True, "data": elem, "bbox_idx": bbox_idx}

            except Exception as e:
                print(f"bbox {bbox_idx} failed (attempt {attempt + 1}): {e}")
                if attempt + 1 < num_attempt:
                    await asyncio.sleep(3)

        return {"success": False, "bbox_idx": bbox_idx}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_with_semaphore(bbox_idx: int):
        async with semaphore:
            return await process_bbox_with_retry(bbox_idx)
    tasks = [_process_with_semaphore(idx) for idx in valid_indices]
    results = []

    for coro in async_tqdm.as_completed(tasks, total=len(tasks)):
        result = await coro
        results.append(result)

    resulted_data = []
    failed_bboxes = []
    
    for result in results:
        if result["success"]:
            resulted_data.append(result["data"])
        else:
            failed_bboxes.append(result["bbox_idx"])

    resulted_data.sort(key=lambda x: x["bbox_idx"])

    if failed_bboxes:
        print(f"Failed to process {len(failed_bboxes)} bboxes: {failed_bboxes}")

    utils.save_json(data=resulted_data, output_path=output_path)
    
    print(f"Successfully processed {len(resulted_data)}/{len(valid_indices)} bboxes")
    return resulted_data


def run_segmentation(
    doc_path: str,
    output_path: str = "/home/lameus/Projects/layout-aware-doc-segmentation/data/nirsii_result.json",
    save_img: bool = True,
    device: str = "cuda",
    use_api: bool = True,
    num_attempt: int = 10,
    max_concurrent: int = 10,
    use_async: bool = True,
    use_batch: bool = False,
    use_context: bool = True
):
    if use_async and use_api:
        batch_processing = (
            run_segmentation_batch_async(
                doc_path=doc_path,
                output_path=output_path,
                save_img=save_img,
                device=device,
                use_api=use_api,
                num_attempt=num_attempt,
                max_concurrent=max_concurrent,
                use_context=use_context,
            )
            if use_batch
            else run_segmentation_async(
                doc_path=doc_path,
                output_path=output_path,
                save_img=save_img,
                device=device,
                use_api=use_api,
                num_attempt=num_attempt,
                max_concurrent=max_concurrent,
                use_async=use_async,
                use_context=use_context,
            )
        )
        return asyncio.run(batch_processing)
    else:
        return run_segmentation_sync(
            doc_path=doc_path,
            output_path=output_path,
            save_img=save_img,
            device=device,
            use_api=use_api,
            num_attempt=num_attempt,
            use_context=use_context,
        )


def run_segmentation_sync(
    doc_path: str,
    output_path: str = "./book.json",
    save_img: bool = False,
    device: str = "cuda",
    use_api: bool = True,
    num_attempt: int = 3,
    use_context: bool = True
):
    layout_extractor = LayoutExtractor(save_img=save_img, device=device)

    bboxes = layout_extractor.get_bboxes_batched(doc_path=doc_path, output_path=output_path)
    pages = layout_extractor.pages

    layout_extractor._find_closest_bboxes()
    layout_extractor._merge_related_bboxes()
    layout_extractor._merge_adjacent_plain_texts(distance_ratio=0.25, abs_gap_px=18)
    layout_extractor._detect_captions_around_figures(distance_ratio=0.1)

    ocr_vlm = ImageDescription(use_api=use_api, pages=pages, bbox_json=layout_extractor.bbox_json,
                               use_context=use_context)
    resulted_data = []

    print("VLM processing")
    for i in tqdm(range(len(layout_extractor.bbox_json))):
        if layout_extractor.bbox_json[i]["ignore"]:
            continue

        attempt = 0
        success = False

        while attempt < num_attempt and not success:
            try:
                vlm_answer = ocr_vlm.inference(bbox_idx=i)
                parsed_answer = ocr_vlm._parse_json(vlm_answer)

                elem = layout_extractor.bbox_json[i].copy()
                elem["description"] = parsed_answer.get("description", "")
                elem["text"] = parsed_answer.get("text", "")
                resulted_data.append(elem)

                if isinstance(elem["image_bytes"], bytes):
                    elem["image_bytes"] = base64.b64encode(elem["image_bytes"]).decode("utf-8")

                success = True

            except json.JSONDecodeError as e:
                print(f"JSON parsing error for bbox {i}, attempt {attempt + 1}: {e}")
                attempt += 1

            except Exception as e:
                print(f"Error processing bbox {i}, attempt {attempt + 1}: {e}")
                attempt += 1

        if not success:
            time.sleep(2)
            print(f"Failed to process bbox {i} after {num_attempt} attempts")

    utils.save_json(data=resulted_data, output_path=output_path)

    return resulted_data


if __name__ == "__main__":
    # async with semaphore 
    # run_segmentation(
    #     doc_path="./part.pdf",
    #     output_path="./data/part_result.json",
    #     use_async=True,
    #     use_batch=False,
    #     max_concurrent=3,
    #     use_context=False,
    # )

    # async with batch processing
    run_segmentation(
        doc_path="./part.pdf",
        output_path="./data/part_result.json",
        use_async=True,
        use_batch=True,
        max_concurrent=10,
        use_context=True,
    )
    