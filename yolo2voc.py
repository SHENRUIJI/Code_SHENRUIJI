import os
import cv2
from xml.dom.minidom import Document

def yolo2voc(img_dir, lbl_dir, xml_out):
    label_map = {'0': "crop", '1': "weed1", '2': "weed2"}
    txt_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]

    for i, fname in enumerate(txt_files):
        img_name = os.path.splitext(fname)[0] + '.jpg'
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(lbl_dir, fname)
        out_path = os.path.join(xml_out, img_name.replace('.jpg', '.xml'))

        if not os.path.exists(img_path):
            print(f"Missing image: {img_name}")
            continue

        img = cv2.imread(img_path)
        h, w, c = img.shape

        doc = Document()
        root = doc.createElement("annotation")
        doc.appendChild(root)

        def append_node(parent, tag, val):
            node = doc.createElement(tag)
            node.appendChild(doc.createTextNode(str(val)))
            parent.appendChild(node)

        append_node(root, "folder", "dataset")
        append_node(root, "filename", img_name)

        size = doc.createElement("size")
        append_node(size, "width", w)
        append_node(size, "height", h)
        append_node(size, "depth", c)
        root.appendChild(size)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = parts
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

                xmin = int((xc - bw / 2) * w)
                ymin = int((yc - bh / 2) * h)
                xmax = int((xc + bw / 2) * w)
                ymax = int((yc + bh / 2) * h)

                obj = doc.createElement("object")
                append_node(obj, "name", label_map.get(cls, "unknown"))
                append_node(obj, "pose", "Unspecified")
                append_node(obj, "truncated", 0)
                append_node(obj, "difficult", 0)

                box = doc.createElement("bndbox")
                append_node(box, "xmin", xmin)
                append_node(box, "ymin", ymin)
                append_node(box, "xmax", xmax)
                append_node(box, "ymax", ymax)
                obj.appendChild(box)

                root.appendChild(obj)

        with open(out_path, 'w', encoding='utf-8') as f:
            doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    img_dir = "D:/Projects/Pycharm_projects/nil/database/data/images/"
    lbl_dir = "D:/Projects/Pycharm_projects/nil/database/data/labels/"
    xml_out = "D:/Projects/Pycharm_projects/nil/database/voc_crop/"
    yolo2voc(img_dir, lbl_dir, xml_out)
