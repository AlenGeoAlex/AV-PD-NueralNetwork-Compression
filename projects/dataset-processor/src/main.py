from src import KITTIConverter


def main():
    print("Hello, world!")
    converter = KITTIConverter(
        images_dir="/Users/alenalex/Work/UCC/S2/scalable/data-set/kitti/data_object_image_2/training/image_2",
        labels_dir="/Users/alenalex/Work/UCC/S2/scalable/data-set/kitti/data_object_label_2/training/label_2",
        output_dir="/Users/alenalex/Work/UCC/S2/scalable/data-set/kitti/normalized_labels",
    )
    converter.convert()

if __name__ == "__main__":
    main()