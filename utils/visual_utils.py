""" Bounding Box Visualization Utility. 

This utilty is used to filter and draw bounding boxes returned by the object detection model.
Reference: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

"""
import cv2
import numpy as np
from matplotlib import colors

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_bounding_box_on_cv2_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str=None,
                               use_normalized_coordinates=False):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: image.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """

  
  (r,g,b,a) = colors.to_rgba(color)
  font = cv2.FONT_HERSHEY_SIMPLEX
  color = (int(r*255),int(g*255),int(b*255))
  im_height,im_width,_ = image.shape

  if use_normalized_coordinates:
    (xmin, xmax, ymin, ymax) = (int(xmin * im_width), int(xmax * im_width),
                                  int(ymin * im_height), int(ymax * im_height))

  #bounding box
  cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color,thickness=thickness)
  #label
  cv2.rectangle(image, (xmin-1, ymin-1),(xmax-1, ymin-10), color, -1)
  #labeltext
  cv2.putText(image,display_str,(xmin,ymin-2), font, 0.22,(0,0,0),1,cv2.LINE_AA)

  return None

def get_or_visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    line_thickness=4,
    color="Yellow",
    visualization_flag=True):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    line_thickness: integer (default: 4) controlling line width of the boxes.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  (ht,wd,_) = image.shape
  normalized_bboxes = []
  to_return_bboxes = []
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      normalized_bboxes.append(box)

      if classes[i] in category_index.keys():
        class_name = category_index[classes[i]]['name']
      else:
        class_name = 'N/A'

      class_score = int(100*scores[i])
      display_str = '{}: {}%'.format(str(class_name), class_score)


  # Draw all boxes onto image.
  for _box in normalized_bboxes:
    ymin, xmin, ymax, xmax = _box
    
    if not use_normalized_coordinates:
      (xmin,ymin,xmax,ymax) = (int(xmin * wd),int(ymin * ht),
                                int(xmax * wd),int(ymax * ht))
      use_normalized_coordinates = False
      centroid = (xmin + int((xmax - xmin)/2) , ymin + int((ymax - ymin)/2))
    else:
      centroid = (xmin + ((xmax - xmin)/2) , ymin + ((ymax - ymin)/2))


    if visualization_flag:
      draw_bounding_box_on_cv2_image(
          image,
          ymin,
          xmin,
          ymax,
          xmax,
          color=color,
          thickness=line_thickness,
          display_str=display_str,
          use_normalized_coordinates=use_normalized_coordinates)

    to_return_bboxes.append((xmin, ymin, xmax, ymax, centroid[0], centroid[1],class_score))

  return image,to_return_bboxes
