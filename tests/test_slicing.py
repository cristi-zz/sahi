# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np
from PIL import Image
from typing import List

from sahi.slicing import slice_coco, slice_image
from sahi.utils.coco import Coco
from sahi.utils.cv import read_image


class TestSlicing(unittest.TestCase):
    def test_slice_image(self):
        # read coco file
        coco_path = "tests/data/coco_utils/terrain1_coco.json"
        coco = Coco.from_coco_dict_or_path(coco_path)

        output_file_name = None
        output_dir = None
        image_path = "tests/data/coco_utils/" + coco.images[0].file_name
        slice_image_result = slice_image(
            image=image_path,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(slice_image_result.images), 18)
        self.assertEqual(len(slice_image_result.coco_images), 18)
        self.assertEqual(slice_image_result.coco_images[0].annotations, [])
        self.assertEqual(slice_image_result.coco_images[15].annotations[1].area, 7296)
        self.assertEqual(
            slice_image_result.coco_images[15].annotations[1].bbox,
            [17, 186, 48, 152],
        )

        image_cv = read_image(image_path)
        slice_image_result = slice_image(
            image=image_cv,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(slice_image_result.images), 18)
        self.assertEqual(len(slice_image_result.coco_images), 18)
        self.assertEqual(slice_image_result.coco_images[0].annotations, [])
        self.assertEqual(slice_image_result.coco_images[15].annotations[1].area, 7296)
        self.assertEqual(
            slice_image_result.coco_images[15].annotations[1].bbox,
            [17, 186, 48, 152],
        )

        image_pil = Image.open(image_path)
        slice_image_result = slice_image(
            image=image_pil,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(slice_image_result.images), 18)
        self.assertEqual(len(slice_image_result.coco_images), 18)
        self.assertEqual(slice_image_result.coco_images[0].annotations, [])
        self.assertEqual(slice_image_result.coco_images[15].annotations[1].area, 7296)
        self.assertEqual(
            slice_image_result.coco_images[15].annotations[1].bbox,
            [17, 186, 48, 152],
        )

    def test_slice_coco(self):
        import shutil

        coco_annotation_file_path = "tests/data/coco_utils/terrain1_coco.json"
        image_dir = "tests/data/coco_utils/"
        output_coco_annotation_file_name = "test_out"
        output_dir = "tests/data/coco_utils/test_out/"
        ignore_negative_samples = True
        coco_dict, _ = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=image_dir,
            output_coco_annotation_file_name=output_coco_annotation_file_name,
            output_dir=output_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(coco_dict["images"]), 5)
        self.assertEqual(coco_dict["images"][1]["height"], 512)
        self.assertEqual(coco_dict["images"][1]["width"], 512)
        self.assertEqual(len(coco_dict["annotations"]), 14)
        self.assertEqual(coco_dict["annotations"][2]["id"], 3)
        self.assertEqual(coco_dict["annotations"][2]["image_id"], 2)
        self.assertEqual(coco_dict["annotations"][2]["category_id"], 1)
        self.assertEqual(coco_dict["annotations"][2]["area"], 12483)
        self.assertEqual(
            coco_dict["annotations"][2]["bbox"],
            [340, 204, 73, 171],
        )

        shutil.rmtree(output_dir, ignore_errors=True)

        coco_annotation_file_path = "tests/data/coco_utils/terrain1_coco.json"
        image_dir = "tests/data/coco_utils/"
        output_coco_annotation_file_name = "test_out"
        output_dir = "tests/data/coco_utils/test_out/"
        ignore_negative_samples = False
        coco_dict, _ = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=image_dir,
            output_coco_annotation_file_name=output_coco_annotation_file_name,
            output_dir=output_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(coco_dict["images"]), 18)
        self.assertEqual(coco_dict["images"][1]["height"], 512)
        self.assertEqual(coco_dict["images"][1]["width"], 512)
        self.assertEqual(len(coco_dict["annotations"]), 14)
        self.assertEqual(coco_dict["annotations"][2]["id"], 3)
        self.assertEqual(coco_dict["annotations"][2]["image_id"], 14)
        self.assertEqual(coco_dict["annotations"][2]["category_id"], 1)
        self.assertEqual(coco_dict["annotations"][2]["area"], 12483)
        self.assertEqual(
            coco_dict["annotations"][2]["bbox"],
            [340, 204, 73, 171],
        )

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_test_slice_coco_with_callbacks(self):
        import shutil

        def custom_slicer(image_height: int,
                          image_width: int,
                          slice_height: int = 512,
                          slice_width: int = 512,
                          overlap_height_ratio: float = 0.2,
                          overlap_width_ratio: float = 0.2,
                          vertical_split_ratio = 0.5
                          ) -> List[List[int]]:
            slice_bboxes = []
            vert_end = int(image_width * vertical_split_ratio)
            left_slice = [0, 0, vert_end, image_height]
            right_slice = [vert_end, 0, image_width, image_height]
            slice_bboxes.append(left_slice)
            slice_bboxes.append(right_slice)
            return slice_bboxes

        coco_annotation_file_path = "data/coco_utils/terrain1_coco.json"
        image_dir = "data/coco_utils/"
        output_coco_annotation_file_name = "test_out"
        output_dir = "data/coco_utils/test_out/"
        ignore_negative_samples = True
        processed_coco_dict, _ = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=image_dir,
            output_coco_annotation_file_name=output_coco_annotation_file_name,
            output_dir=output_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
            bbox_generator=custom_slicer,
        )

        self.assertEqual(len(processed_coco_dict["images"]), 2)
        total_width = processed_coco_dict["images"][0]["width"] +\
                      processed_coco_dict["images"][1]["width"]
        self.assertEqual(total_width, 2048)
        # self.assertEqual(processed_coco_dict["images"][1]["height"], 512)
        # self.assertEqual(processed_coco_dict["images"][1]["width"], 512)
        # self.assertEqual(len(processed_coco_dict["annotations"]), 14)
        # self.assertEqual(processed_coco_dict["annotations"][2]["id"], 3)
        # self.assertEqual(processed_coco_dict["annotations"][2]["image_id"], 2)
        # self.assertEqual(processed_coco_dict["annotations"][2]["category_id"], 1)
        # self.assertEqual(processed_coco_dict["annotations"][2]["area"], 12483)
        # self.assertEqual(
        #     processed_coco_dict["annotations"][2]["bbox"],
        #     [340, 204, 73, 171],
        # )

        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
