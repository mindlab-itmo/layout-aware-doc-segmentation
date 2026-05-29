import asyncio
import base64
import json
import time

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from src.doc_layout import LayoutExtractor
from src.doc_ocr import ImageDescription


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
        """Process single bbox with retry logic"""
        attempt = 0
        
        while attempt < num_attempt:
            try:
                vlm_answer = await ocr_vlm.inference_async(bbox_idx=bbox_idx)
                parsed_answer = ocr_vlm._parse_json(vlm_answer)

                elem = layout_extractor.bbox_json[bbox_idx].copy()
                elem["description"] = parsed_answer.get("description", "")
                elem["text"] = parsed_answer.get("text", "")

                if isinstance(elem["image_bytes"], bytes):
                    elem["image_bytes"] = base64.b64encode(elem["image_bytes"]).decode("utf-8")
                
                return {"success": True, "data": elem, "bbox_idx": bbox_idx}
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for bbox {bbox_idx}, attempt {attempt + 1}: {e}")
                attempt += 1
                await asyncio.sleep(1)
                if attempt < num_attempt:
                    await asyncio.sleep(3)
                
            except Exception as e:
                print(f"Error processing bbox {bbox_idx}, attempt {attempt + 1}: {e}")
                attempt += 1
                await asyncio.sleep(1)
                if attempt < num_attempt:
                    await asyncio.sleep(3)

        return {"success": False, "bbox_idx": bbox_idx}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(bbox_idx: int):
        async with semaphore:
            return await process_bbox_with_retry(bbox_idx)
    tasks = [process_with_semaphore(idx) for idx in valid_indices]
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

    # Save results
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(resulted_data, f, ensure_ascii=False, indent=2)
    
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
    use_context: bool = True
):  
    if use_async and use_api:
        # Run async version
        return asyncio.run(
            run_segmentation_async(
                doc_path=doc_path,
                output_path=output_path,
                save_img=save_img,
                device=device,
                use_api=use_api,
                num_attempt=num_attempt,
                max_concurrent=max_concurrent,
                use_async=use_async,
                use_context=use_context
            )
        )
    else:
        # Run sync version
        return run_segmentation_sync(
            doc_path=doc_path,
            output_path=output_path,
            save_img=save_img,
            device=device,
            use_api=use_api,
            num_attempt=num_attempt,
            use_context=use_context
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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resulted_data, f, ensure_ascii=False, indent=2)

    return resulted_data


if __name__ == "__main__":
    run_segmentation(
        doc_path="./part.pdf",
        output_path="./data/part_result.json",
        use_async=True,
        max_concurrent=3,
        use_context=False  
    )
    