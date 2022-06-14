from datasets import CityScapesDataset
import models
import torch
import train_test
from torch import nn

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    json_path = "/media/susi/Elements/Datasets/cityscapes/CityScapes_Original/gtFine_trainvaltest/gtFine/train/aachen"
    img_path = "/media/susi/Elements/Datasets/cityscapes/CityScapes_Original/leftImg8bit_trainvaltest/leftImg8bit/train/aachen"

    dataset = CityScapesDataset(json_dir=json_path, img_dir=img_path, classes=['road', 'car'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    num_classes = 3
    model = models.get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        train_test.train(data_loader, model, loss_fn, optimizer, device)
        print("poch:"+str(epoch))




