import os
import xml.etree.ElementTree as ET
import cv2

# Define the mapping of class names to YOLO class IDs
class_mapping = {
    'pug': 0,
    'boxer': 1,
    # Add more class mappings as needed
}

# Function to get image dimensions
def get_image_dimensions(image_dir, xml_filename):
    image_filename = os.path.splitext(xml_filename)[0] + '.jpg'
    image_path = os.path.join(image_dir, image_filename)

    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            height, width, _ = image.shape
            return width, height
    return None, None

# Function to convert Pascal VOC XML to YOLO TXT with normalized coordinates
def convert_pascal_voc_to_yolo(input_dir, output_dir, image_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(''):
            xml_path = os.path.join(input_dir, filename)
            yolo_txt_filename = os.path.splitext(filename)[0] + '.txt'
            yolo_txt_path = os.path.join(output_dir, yolo_txt_filename)

            image_width, image_height = get_image_dimensions(image_dir, filename)

            if image_width is None or image_height is None:
                print(f"Warning: Image dimensions not found for {filename}. Skipping...")
                continue

            with open(yolo_txt_path, 'w') as yolo_file:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in class_mapping:
                        print(f"Warning: Class '{class_name}' not found in class mapping. Skipping...")
                        continue

                    class_id = class_mapping[class_name]

                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    # Normalize the coordinates
                    center_x = (xmin + xmax) / (2.0 * image_width)
                    center_y = (ymin + ymax) / (2.0 * image_height)
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height

                    # YOLO format: class_id center_x center_y width height
                    yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                    yolo_file.write(yolo_line)

            print(f"Converted: {xml_path} -> {yolo_txt_path}")

if __name__ == "__main__":
    input_dir = "data\\pug\\annotations\\xml-annotations"
    output_dir = "data\\pug\\annotations\\n02110958-pug"
    image_dir = "data/pug/images/n02110958-pug"
    convert_pascal_voc_to_yolo(input_dir, output_dir, image_dir)
