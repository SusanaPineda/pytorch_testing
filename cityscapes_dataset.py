from datasets import CityScapesDataset, dataLoaderCitysCapes
import models
import torch
from torch import nn
from engine import train_one_epoch, evaluate

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    json_path = "data/cityscapes/gtFine_trainvaltest/gtFine/train/aachen"
    img_path = "data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen"

    dataset = CityScapesDataset(json_dir=json_path, img_dir=img_path, classes=['road', 'car'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataLoaderCitysCapes)

    json_path = "data/cityscapes/gtFine_trainvaltest/gtFine/train/bochum"
    img_path = "data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/bochum"

    dataset_test = CityScapesDataset(json_dir=json_path, img_dir=img_path, classes=['road', 'car'])
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=dataLoaderCitysCapes)

    num_classes = 3
    model = models.get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                 lr=0.0001,
                                momentum=0.9,
                                 weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 5

    for epoch in range(num_epochs):
        # train_test.train(data_loader, model, loss_fn, optimizer, device)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)




