from __future__ import annotations

import gc
from typing import Optional, Union
import math

import numpy as np
from ncnn_vulkan import ncnn
from sanic.log import logger

from .auto_split import Split
from .utils import get_h_w_c


def fix_dtype_range(img):
    dtype_max = 1
    try:
        dtype_max = np.iinfo(img.dtype).max
    except:
        logger.debug("img dtype is not an int")

    img = (
        (np.clip(img.astype("float32") / dtype_max, 0, 1) * 255)
        .round()
        .astype(np.uint8)
    )
    return img


def upscale(
    img: np.ndarray,
    net,
    input_name: str,
    output_name: str,
    blob_vkallocator,
    staging_vkallocator,
):
    ex = net.create_extractor()
    ex.set_blob_vkallocator(blob_vkallocator)
    ex.set_workspace_vkallocator(blob_vkallocator)
    ex.set_staging_vkallocator(staging_vkallocator)
    # ex.set_light_mode(True)
    try:
        lr_c = get_h_w_c(img)[2]
        lr_img_fix = fix_dtype_range(img)
        if lr_c == 1:
            pixel_type = ncnn.Mat.PixelType.PIXEL_GRAY
        elif lr_c == 3:
            pixel_type = ncnn.Mat.PixelType.PIXEL_RGB
        else:
            pixel_type = ncnn.Mat.PixelType.PIXEL_RGBA
        mat_in = ncnn.Mat.from_pixels(
            lr_img_fix,
            pixel_type,
            lr_img_fix.shape[1],
            lr_img_fix.shape[0],
        )
        mean_vals = []
        norm_vals = [1 / 255.0] * lr_c
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        ex.input(input_name, mat_in)
        _, mat_out = ex.extract(output_name)
        result = np.array(mat_out).transpose(1, 2, 0).astype(np.float32)
        del ex, mat_in, mat_out
        gc.collect()
        # Clear VRAM
        blob_vkallocator.clear()
        staging_vkallocator.clear()
        return result
    except Exception as e:
        if "vkQueueSubmit" in str(e):
            ex = None
            del ex
            gc.collect()
            blob_vkallocator.clear()
            staging_vkallocator.clear()
            # TODO: Have someone running into this issue enable this and see if it fixes anything
            # ncnn.destroy_gpu_instance()
            raise RuntimeError(
                "A critical error has occurred. You may need to restart chaiNNer in order for NCNN upscaling to start working again."
            ) from e
        # Check to see if its actually the NCNN out of memory error
        if "failed" in str(e):
            # clear VRAM
            logger.info(f"NCNN out of VRAM, clearing VRAM and splitting.")
            ex = None
            del ex
            gc.collect()
            blob_vkallocator.clear()
            staging_vkallocator.clear()
            return Split()
        else:
            # Re-raise the exception if not an OOM error
            raise


def auto_split(
    img: np.ndarray,
    net,
    input_name: str,
    output_name: str,
    blob_vkallocator,
    staging_vkallocator,
    max_tile_size: Union[int, None] = None,
    overlap: int = 16,
    min_tile_size: int = 16,
) -> np.ndarray:
    """
    Splits the image into tiles with at most the given tile size.

    If the upscale method requests a split, then the tile size will be lowered.
    """

    h, w, c = get_h_w_c(img)

    if max_tile_size is None:
        max_tile_size = max(256, int(w * 1.05), int(h * 1.05))
        logger.info(
            f"Auto split image ({w}x{h}px @ {c}) with initial tile size defaulting to {max_tile_size}."
        )
    else:
        logger.info(
            f"Auto split image ({w}x{h}px @ {c}) with initial tile size {max_tile_size}."
        )

    if h <= max_tile_size and w <= max_tile_size:
        # the image might be small enough so that we don't have to split at all
        upscale_result = upscale(
            img, net, input_name, output_name, blob_vkallocator, staging_vkallocator
        )
        if not isinstance(upscale_result, Split):
            return upscale_result

        # the image was too large
        max_tile_size = max_tile_size // 2
        while max_tile_size * max_tile_size > w * h:
            max_tile_size = max_tile_size // 2

        logger.info(
            f"Unable to upscale the whole image at once. Reduced tile size to {max_tile_size}."
        )

    # The upscale method is allowed to request splits at any time.
    # When a split occurs, we have to "restart" the loop and
    # these 2 variables allow us to split the already processed tiles.
    start_x = 0
    start_y = 0

    # To allocate the result image, we need to know the upscale factor first,
    # and we only get to know this factor after the first successful upscale.
    result: Optional[np.ndarray] = None
    scale: int = 0

    restart = True
    while restart:
        restart = False

        assert max_tile_size >= min_tile_size

        # This is a bit complex.
        # We don't actually use the current tile size to partition the image.
        # If we did, then tile_size=1024 and w=1200 would result in very uneven tiles.
        # Instead, we use tile_size to calculate how many tiles we get in the x and y direction
        # and then calculate the optimal tile size for the x and y direction using the counts.
        # This yields optimal tile sizes which should prevent unnecessary splitting.
        tile_count_x = math.ceil(w / max_tile_size)
        tile_count_y = math.ceil(h / max_tile_size)
        tile_size_x = math.ceil(w / tile_count_x)
        tile_size_y = math.ceil(h / tile_count_y)

        logger.info(
            f"Currently {tile_count_x}x{tile_count_y} tiles each {tile_size_x}x{tile_size_y}px."
        )

        for y in range(0, tile_count_y):
            if restart:
                break
            if y < start_y:
                continue

            for x in range(0, tile_count_x):
                if y == start_y and x < start_x:
                    continue

                x_min = max(0, x * tile_size_x - overlap)
                y_min = max(0, y * tile_size_y - overlap)
                x_max = min(w, (x + 1) * tile_size_x + overlap)
                y_max = min(h, (y + 1) * tile_size_y + overlap)

                upscale_result = upscale(
                    img[y_min:y_max, x_min:x_max, ...],
                    net,
                    input_name,
                    output_name,
                    blob_vkallocator,
                    staging_vkallocator,
                )

                if isinstance(upscale_result, Split):
                    max_tile_size = max_tile_size // 2

                    new_tile_count_x = math.ceil(w / max_tile_size)
                    new_tile_count_y = math.ceil(h / max_tile_size)
                    new_tile_size_x = math.ceil(w / new_tile_count_x)
                    new_tile_size_y = math.ceil(h / new_tile_count_y)
                    start_x = (x * tile_size_x) // new_tile_size_x
                    start_y = (y * tile_size_x) // new_tile_size_y

                    logger.info(
                        f"Split occurred. New tile size is {max_tile_size}. Starting at {start_x},{start_y}."
                    )

                    restart = True
                    break

                # figure out by how much the image was upscaled by
                up_h, up_w, _ = get_h_w_c(upscale_result)
                current_scale = up_h // (y_max - y_min)
                assert current_scale > 0
                assert (y_max - y_min) * current_scale == up_h
                assert (x_max - x_min) * current_scale == up_w

                if result is None:
                    # allocate the result image
                    scale = current_scale
                    result = np.zeros((h * scale, w * scale, c), dtype=np.float32)

                assert current_scale == scale

                # remove overlap padding
                pad_left = abs(x * tile_size_x - x_min)
                pad_top = abs(y * tile_size_y - y_min)
                pad_right = abs(min(w, (x + 1) * tile_size_x) - x_max)
                pad_bottom = abs(min(h, (y + 1) * tile_size_y) - y_max)

                up_x = pad_left * scale
                up_y = pad_top * scale
                up_w = up_w - (pad_left + pad_right) * scale
                up_h = up_h - (pad_top + pad_bottom) * scale

                upscale_result = upscale_result[
                    up_y : (up_y + up_h),
                    up_x : (up_x + up_w),
                    ...,
                ]

                # copy into result image
                res_x = x * tile_size_x * scale
                res_y = y * tile_size_y * scale
                result[
                    res_y : (res_y + up_h),
                    res_x : (res_x + up_w),
                    ...,
                ] = upscale_result

    assert result is not None
    return result


def ncnn_auto_split(
    img: np.ndarray,
    net,
    input_name: str,
    output_name: str,
    blob_vkallocator,
    staging_vkallocator,
    max_tile_size: Union[int, None] = None,
) -> np.ndarray:
    return auto_split(
        img,
        net,
        input_name,
        output_name,
        blob_vkallocator,
        staging_vkallocator,
        max_tile_size,
    )
