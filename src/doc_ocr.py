import io
import re
import cv2
import json
import base64
import numpy as np
from dotenv import load_dotenv
import asyncio
from typing import List, Optional, Union

import torch
from openai import OpenAI, AsyncOpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from src.utils import utils

load_dotenv()


class ImageDescription:
    def __init__(
        self,
        use_api: bool = True,
        pages: list = [],
        bbox_json: list = None,
        device: str = "cuda",
        model_name: str = "qwen3.6:35b-a3b-128k",
        use_async: bool = False,
        use_context: bool = True
    ) -> None:
        """Class for LLM usage: it uses either the results
        of doc_layout or it can process a file from scratch.

        Args:
            use_api (bool, optional): an argument to define a mode of LLM: True to use it with API
            False for local usage. Defaults to True.
            bbox_json (list, optional): an output from doc_layout. Defaults to None.
            device (str, optional): device type "cpu"/"cuda". Defaults to "cuda".
            model_name (str, optional): model name/path to the model. Defaults to "Qwen/Qwen2.5-VL-7B-Instruct".
            use_async (bool, optional): whether to use async API calls. Defaults to False.
        """

        self.use_api = use_api
        self.use_async = use_async
        self.use_context = use_context
        self.pages = pages
        self.model_name = model_name
        self.device = device

        # obtained with doc_layout
        self.bbox_json = bbox_json

        if self.use_api:
            if self.use_async:
                self.model = AsyncOpenAI()
            else:
                self.model = OpenAI()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, device_map=device
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.prompt = (
            "Describe the document fragment in detail. "
            "Use the context of the full page that the fragment belongs to. "
            'Respond with ONLY a JSON object — no markdown, no extra text — '
            'with exactly two keys: "description" (string) and "text" (string).'
        )

    def _parse_json(self, json_output):
        """Response postprocessing"""
        text = json_output.strip()

        if "```" in text:
            for fence in ("```json", "```"):
                if fence in text:
                    parts = text.split(fence)
                    for part in parts[1:]:
                        candidate = part.split("```")[0].strip()
                        if candidate:
                            text = candidate
                            break
                    break
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError(f"Could not parse JSON from LLM output: {json_output[:200]!r}")

    def load_doc(self, doc_path):
        self.image = cv2.imread(doc_path)

    def readb64(self, uri):
        nparr = np.frombuffer(base64.b64decode(uri), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def _prepare_image_data(
        self, bbox_idx: Optional[int] = None, bounding_box=None
    ) -> str:
        """Prepare base64 encoded image data from bbox_idx or bounding_box"""
        if bbox_idx is not None:
            cropped_img_byte = self.bbox_json[bbox_idx]["image_bytes"]

            if isinstance(cropped_img_byte, str):
                cropped_img_byte = eval(cropped_img_byte)

            if isinstance(cropped_img_byte, bytes):
                cropped_img_byte = cropped_img_byte.decode("utf-8")

        elif bounding_box != None:
            x, y, w, h = [int(i) for i in bounding_box]
            cropped_img = self.image[y : y + h, x : x + w]
            retval, buffer = cv2.imencode(".jpg", cropped_img)
            cropped_img_byte = base64.b64encode(buffer).decode("utf-8")
        else:
            raise NotImplementedError(
                "Should be selected bbox_idx or bounding_box as input"
            )
        return cropped_img_byte

    def _prepare_context_image(
        self, bbox_idx: int, downsample_factor: int = 6
    ) -> Optional[str]:
        """Prepare base64 encoded context image (full page downsampled)"""
        if len(self.pages) == 0 or bbox_idx is None:
            return None

        full_page = self.pages[self.bbox_json[bbox_idx]["page_num"]]

        if not isinstance(full_page, np.ndarray):
            raise TypeError(f"Expected numpy array for page, got {type(full_page)}")

        full_page_rgb = cv2.cvtColor(full_page, cv2.COLOR_BGR2RGB)

        context_img = utils.downsample_image(full_page_rgb, downsample_factor)

        context_buffer = io.BytesIO()
        context_img.save(context_buffer, format="JPEG", quality=60)
        context_bytes = context_buffer.getvalue()
        context_img_bytes = base64.b64encode(context_bytes).decode("utf-8")

        return context_img_bytes

    def _build_messages(
        self,
        cropped_img_byte: str,
        sys_prompt: str,
        context_img_byte: Optional[str] = None,
    ) -> List[dict]:
        """Build messages for API call with optional context image"""
        content = [
            {"type": "text", "text": self.prompt},
        ]

        if context_img_byte is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{context_img_byte}"},
                }
            )

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{cropped_img_byte}"},
            }
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": content,
            },
        ]
        return messages

    async def inference_async(
        self,
        bbox_idx: int = None,
        bounding_box=None,
        sys_prompt="You are a technical document specialist.",
        max_new_tokens=2048,
        downsample_factor: int = 6,
    ) -> str:
        """Async inference method for API calls

        Args:
            bbox_idx: Index of bbox in bbox_json
            bounding_box: Alternative to bbox_idx - direct bounding box coordinates
            sys_prompt: System prompt for the model
            max_new_tokens: Maximum tokens to generate
            downsample_factor: Factor to downsample context image
        """
        if not self.use_api or not self.use_async:
            raise ValueError(
                "Async inference only available with use_api=True and use_async=True"
            )

        cropped_img_byte = self._prepare_image_data(bbox_idx, bounding_box)

        context_img_byte = None
        if self.use_context and bbox_idx is not None:
            name = self.bbox_json[bbox_idx].get("name", "")
            if name not in ("plain text", "title"):
                context_img_byte = self._prepare_context_image(bbox_idx, downsample_factor)

        messages = self._build_messages(cropped_img_byte, sys_prompt, context_img_byte)

        try:
            response = await self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            choice = response.choices[0]
            output_text = choice.message.content
            if not output_text:
                raise ValueError(
                    f"Empty LLM content for bbox_idx={bbox_idx}; "
                    f"finish_reason={choice.finish_reason!r}"
                )
            return output_text
        except Exception as e:
            print(f"Async API call failed for bbox_idx {bbox_idx}: {e}")
            raise

    async def batch_inference_async(
        self,
        bbox_indices: List[int] = None,
        bounding_boxes: List = None,
        sys_prompt="You are a technical document specialist.",
        max_new_tokens=2048,
        max_concurrent: int = 10,
        downsample_factor: int = 6,
    ) -> List[str]:
        """Process multiple bboxes concurrently with rate limiting

        Args:
            bbox_indices: List of bbox indices to process
            bounding_boxes: Alternative - list of bounding box coordinates
            sys_prompt: System prompt for the model
            max_new_tokens: Maximum tokens to generate
            max_concurrent: Maximum concurrent API calls
            downsample_factor: Factor to downsample context images
        """
        if bbox_indices is None and bounding_boxes is None:
            raise ValueError("Must provide either bbox_indices or bounding_boxes")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(bbox_idx=None, bounding_box=None):
            async with semaphore:
                try:
                    result = await self.inference_async(
                        bbox_idx=bbox_idx,
                        bounding_box=bounding_box,
                        sys_prompt=sys_prompt,
                        max_new_tokens=max_new_tokens,
                        downsample_factor=downsample_factor,
                    )
                    return {"success": True, "result": result, "bbox_idx": bbox_idx}
                except Exception as e:
                    return {"success": False, "error": str(e), "bbox_idx": bbox_idx}

        # Create tasks for async
        tasks = []
        if bbox_indices:
            tasks = [process_with_semaphore(bbox_idx=idx) for idx in bbox_indices]
        else:
            tasks = [
                process_with_semaphore(bounding_box=bbox) for bbox in bounding_boxes
            ]

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    def inference(
        self,
        bbox_idx: int = None,
        bounding_box=None,
        sys_prompt="You are a technical document specialist.",
        max_new_tokens=2048,
        return_input=False,
        downsample_factor: int = 6,
    ):
        """The main method to run document with LLM (synchronous).

        Args:
            bbox_idx (int, optional): an option to process specific bbox from bbox_json. Defaults to None.
            bounding_box (_type_, optional): an option to process the area of a document. Defaults to None.
            sys_prompt (str, optional): system prompt. Defaults to "You are a helpful assistant.".
            max_new_tokens (int, optional): _. Defaults to 1024.
            return_input (bool, optional): Option to define return result of processor or not,
            works only for local model. Defaults to False.
            downsample_factor (int, optional): Factor to downsample context image. Defaults to 6.
        Raises:
            NotImplementedError: "Should be selected bbox_idx or bounding_box as input"

        Returns:
            str: a VLM answer {'description': str,
            'text': str}. It's considered to process the output with _parse_json method of the class.
        """
        cropped_img_byte = self._prepare_image_data(bbox_idx, bounding_box)

        if self.use_api:
            context_img_byte = None
            if self.use_context and bbox_idx is not None:
                name = self.bbox_json[bbox_idx].get("name", "")
                if name not in ("plain text", "title"):
                    context_img_byte = self._prepare_context_image(
                        bbox_idx, downsample_factor
                    )

            messages = self._build_messages(
                cropped_img_byte, sys_prompt, context_img_byte
            )

            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                )
                choice = response.choices[0]
                output_text = choice.message.content
                if not output_text:
                    raise ValueError(
                        f"Empty LLM content for bbox_idx={bbox_idx}; "
                        f"finish_reason={choice.finish_reason!r}"
                    )
                return output_text
            except Exception as e:
                print(f"API call failed: {e}")
                raise
        else:
            if bbox_idx is not None:
                cropped_img = self.readb64(cropped_img_byte)
            else:
                # Use the cropped image from bounding_box
                pass

            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"image": cropped_img},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[cropped_img], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            if return_input:
                return output_text[0], inputs
            else:
                return output_text[0]

    async def process_all_bboxes_async(
        self,
        sys_prompt="You are a technical document specialist.",
        max_new_tokens=2048,
        max_concurrent=10,
        filter_ignored=True,
        downsample_factor=6,
    ) -> List[dict]:
        """Process all bboxes in bbox_json asynchronously

        Args:
            sys_prompt: System prompt for the model
            max_new_tokens: Maximum tokens to generate
            max_concurrent: Maximum concurrent API calls
            filter_ignored: Whether to filter out bboxes marked as ignored
            downsample_factor: Factor to downsample context images
        """
        if self.bbox_json is None:
            raise ValueError("bbox_json is not set")

        if filter_ignored:
            bbox_indices = [
                i
                for i, bbox in enumerate(self.bbox_json)
                if not bbox.get("ignore", False)
            ]
        else:
            bbox_indices = list(range(len(self.bbox_json)))

        results = await self.batch_inference_async(
            bbox_indices=bbox_indices,
            sys_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            max_concurrent=max_concurrent,
            downsample_factor=downsample_factor,
        )

        return results
