# adopted from https://github.com/JaidedAI/EasyOCR
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from craft import CRAFT
from collections import OrderedDict
from skimage import io
from craft_utils import adjustResultCoordinates, getDetBoxes
import os


def normalizeMeanVariance(
  in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
  # should be RGB order
  img = in_img.copy().astype(np.float32)

  img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
  img /= np.array(
    [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
    dtype=np.float32,
  )
  return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
  height, width, channel = img.shape
  # magnify image size
  target_size = mag_ratio * max(height, width)

  # set original image size
  if target_size > square_size:
    target_size = square_size

  ratio = target_size / max(height, width)

  target_h, target_w = int(height * ratio), int(width * ratio)
  proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

  # make canvas and paste image
  target_h32, target_w32 = target_h, target_w
  if target_h % 32 != 0:
    target_h32 = target_h + (32 - target_h % 32)
  if target_w % 32 != 0:
    target_w32 = target_w + (32 - target_w % 32)
  resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
  resized[0:target_h, 0:target_w, :] = proc
  target_h, target_w = target_h32, target_w32

  size_heatmap = (int(target_w / 2), int(target_h / 2))

  return resized, ratio, size_heatmap


def test_net(
  canvas_size,
  mag_ratio,
  net,
  image,
  text_threshold,
  link_threshold,
  low_text,
  poly,
  device,
  estimate_num_chars=False,
):
  if (
    isinstance(image, np.ndarray) and len(image.shape) == 4
  ):  # image is batch of np arrays
    image_arrs = image
  else:  # image is single numpy array
    image_arrs = [image]

  img_resized_list = []
  # resize
  for img in image_arrs:
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
      img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    img_resized_list.append(img_resized)
  ratio_h = ratio_w = 1 / target_ratio
  # preprocessing
  x = [
    np.transpose(normalizeMeanVariance(n_img), (2, 0, 1)) for n_img in img_resized_list
  ]
  x = torch.from_numpy(np.array(x))
  x = x.to(device)

  # forward pass
  with torch.no_grad():
    y, feature = net(x)

  boxes_list, polys_list = [], []
  for out in y:
    # make score and link map
    score_text = out[:, :, 0].cpu().data.numpy()
    score_link = out[:, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys, mapper = getDetBoxes(
      score_text,
      score_link,
      text_threshold,
      link_threshold,
      low_text,
      poly,
      estimate_num_chars,
    )

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    if estimate_num_chars:
      boxes = list(boxes)
      polys = list(polys)
    for k in range(len(polys)):
      if estimate_num_chars:
        boxes[k] = (boxes[k], mapper[k])
      if polys[k] is None:
        polys[k] = boxes[k]
    boxes_list.append(boxes)
    polys_list.append(polys)

  return boxes_list, polys_list


def load_image(img_file):
  img = io.imread(img_file)  # RGB order
  if img.shape[0] == 2:
    img = img[0]
  if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  if img.shape[2] == 4:
    img = img[:, :, :3]
  img = np.array(img)
  return img


def copyStateDict(state_dict):
  if list(state_dict.keys())[0].startswith("module"):
    start_idx = 1
  else:
    start_idx = 0
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = ".".join(k.split(".")[start_idx:])
    new_state_dict[name] = v
  return new_state_dict


def get_textbox(
  detector,
  image,
  canvas_size,
  mag_ratio,
  text_threshold,
  link_threshold,
  low_text,
  poly,
  device,
  optimal_num_chars=None,
  **kwargs,
):
  result = []
  estimate_num_chars = optimal_num_chars is not None
  bboxes_list, polys_list = test_net(
    canvas_size,
    mag_ratio,
    detector,
    image,
    text_threshold,
    link_threshold,
    low_text,
    poly,
    device,
    estimate_num_chars,
  )
  if estimate_num_chars:
    polys_list = [
      [p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
      for polys in polys_list
    ]

  for polys in polys_list:
    single_img_result = []
    for i, box in enumerate(polys):
      poly = np.array(box).astype(np.int32).reshape((-1))
      single_img_result.append(poly)
    result.append(single_img_result)

  return result


def diff(input_list):
  return max(input_list) - min(input_list)


def group_text_box(
  polys,
  slope_ths=0.1,
  ycenter_ths=0.5,
  height_ths=0.5,
  width_ths=1.0,
  add_margin=0.05,
  sort_output=True,
):
  # poly top-left, top-right, low-right, low-left
  horizontal_list, free_list, combined_list, merged_list = [], [], [], []

  for poly in polys:
    slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
    slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
    if max(abs(slope_up), abs(slope_down)) < slope_ths:
      x_max = max([poly[0], poly[2], poly[4], poly[6]])
      x_min = min([poly[0], poly[2], poly[4], poly[6]])
      y_max = max([poly[1], poly[3], poly[5], poly[7]])
      y_min = min([poly[1], poly[3], poly[5], poly[7]])
      horizontal_list.append(
        [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min]
      )
    else:
      height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
      width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])

      margin = int(1.44 * add_margin * min(width, height))

      theta13 = abs(
        np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4])))
      )
      theta24 = abs(
        np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6])))
      )
      # do I need to clip minimum, maximum value here?
      x1 = poly[0] - np.cos(theta13) * margin
      y1 = poly[1] - np.sin(theta13) * margin
      x2 = poly[2] + np.cos(theta24) * margin
      y2 = poly[3] - np.sin(theta24) * margin
      x3 = poly[4] + np.cos(theta13) * margin
      y3 = poly[5] + np.sin(theta13) * margin
      x4 = poly[6] - np.cos(theta24) * margin
      y4 = poly[7] + np.sin(theta24) * margin

      free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
  if sort_output:
    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

  # combine box
  new_box = []
  for poly in horizontal_list:
    if len(new_box) == 0:
      b_height = [poly[5]]
      b_ycenter = [poly[4]]
      new_box.append(poly)
    else:
      # comparable height and comparable y_center level up to ths*height
      if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
        b_height.append(poly[5])
        b_ycenter.append(poly[4])
        new_box.append(poly)
      else:
        b_height = [poly[5]]
        b_ycenter = [poly[4]]
        combined_list.append(new_box)
        new_box = [poly]
  combined_list.append(new_box)

  # merge list use sort again
  for boxes in combined_list:
    if len(boxes) == 1:  # one box per line
      box = boxes[0]
      margin = int(add_margin * min(box[1] - box[0], box[5]))
      merged_list.append(
        [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]
      )
    else:  # multiple boxes per line
      boxes = sorted(boxes, key=lambda item: item[0])

      merged_box, new_box = [], []
      for box in boxes:
        if len(new_box) == 0:
          b_height = [box[5]]
          x_max = box[1]
          new_box.append(box)
        else:
          if (abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)) and (
            (box[0] - x_max) < width_ths * (box[3] - box[2])
          ):  # merge boxes
            b_height.append(box[5])
            x_max = box[1]
            new_box.append(box)
          else:
            b_height = [box[5]]
            x_max = box[1]
            merged_box.append(new_box)
            new_box = [box]
      if len(new_box) > 0:
        merged_box.append(new_box)

      for mbox in merged_box:
        if len(mbox) != 1:  # adjacent box in same line
          # do I need to add margin here?
          x_min = min(mbox, key=lambda x: x[0])[0]
          x_max = max(mbox, key=lambda x: x[1])[1]
          y_min = min(mbox, key=lambda x: x[2])[2]
          y_max = max(mbox, key=lambda x: x[3])[3]

          box_width = x_max - x_min
          box_height = y_max - y_min
          margin = int(add_margin * (min(box_width, box_height)))

          merged_list.append(
            [x_min - margin, x_max + margin, y_min - margin, y_max + margin]
          )
        else:  # non adjacent box in same line
          box = mbox[0]

          box_width = box[1] - box[0]
          box_height = box[3] - box[2]
          margin = int(add_margin * (min(box_width, box_height)))

          merged_list.append(
            [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]
          )
  # may need to check if box is really in image
  return merged_list, free_list


def read_text(
  image_file,
  detector,
  device,
  min_size=20,
  text_threshold=0.7,
  low_text=0.3,  # 0.4 before
  link_threshold=0.4,
  canvas_size=2560,
  mag_ratio=1.0,
  slope_ths=0.1,
  ycenter_ths=0.5,
  height_ths=0.5,
  width_ths=0.5,
  add_margin=0.1,
  optimal_num_chars=None,
  threshold=0.2,
  bbox_min_score=0.2,
  bbox_min_size=3,
  max_candidates=0,
):
  img = load_image(image_file)
  text_box_list = get_textbox(
    detector,
    img,
    canvas_size=canvas_size,
    mag_ratio=mag_ratio,
    text_threshold=text_threshold,
    link_threshold=link_threshold,
    low_text=low_text,
    poly=False,
    device=device,
    optimal_num_chars=optimal_num_chars,
    threshold=threshold,
    bbox_min_score=bbox_min_score,
    bbox_min_size=bbox_min_size,
    max_candidates=max_candidates,
  )

  horizontal_list_agg, free_list_agg = [], []
  for text_box in text_box_list:
    horizontal_list, free_list = group_text_box(
      text_box,
      slope_ths,
      ycenter_ths,
      height_ths,
      width_ths,
      add_margin,
      (optimal_num_chars is None),
    )
    if min_size:
      horizontal_list = [
        i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > min_size
      ]
      free_list = [
        i
        for i in free_list
        if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size
      ]
    horizontal_list_agg.append(horizontal_list)
    free_list_agg.append(free_list)
  return horizontal_list_agg, free_list_agg


if __name__ == "__main__":
  image_file = "assets/khmer.jpg"
  device = "cpu"
  trained_model = "assets/craft_mlt_25k.pth"

  net = CRAFT()
  net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
  net.eval()

  horizontal_list_agg, free_list_agg = read_text(image_file, net, device=device)
  image = Image.open(image_file)
  image_draw = ImageDraw.Draw(image)

  # Create a directory to save cropped images
  output_dir = "assets/cropped_text"
  os.makedirs(output_dir, exist_ok=True)

  # Crop and save individual text boxes
  for i, bbox in enumerate(horizontal_list_agg[0]):
      x_min, x_max, y_min, y_max = bbox
      # Crop the image using the bounding box
      cropped_image = image.crop((x_min, y_min, x_max, y_max))
      # Save the cropped image
      cropped_image.save(os.path.join(output_dir, f"text_box_{i + 1}.jpg"))
      image_draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)
      image.save("assets/khmer_output.jpg")


