from datasets import CityScapesDataset

if __name__ == '__main__':
    json_path = "/media/susi/Elements/Datasets/cityscapes/CityScapes_Original/gtFine_trainvaltest/gtFine/train/aachen"
    img_path = "/media/susi/Elements/Datasets/cityscapes/CityScapes_Original/leftImg8bit_trainvaltest/leftImg8bit/train/aachen"

    dataset = CityScapesDataset (json_dir=json_path, img_dir=img_path, classes=['road', 'car'])

