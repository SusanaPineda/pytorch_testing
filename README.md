# Guía de Pytorch

## Instalación:
https://pytorch.org/get-started/locally/

## Tutoriales:
https://pytorch.org/tutorials/beginner/basics/intro.html

### Quickstart
#### Datos:
torch.utils.data.Dataset --> almacena los datos y las etiquetas
torch.utils.data.DataLoader --> crea un iterable sobre Dataset

Los datasets disponibles en torch están formados por datos del tipo Dataset. Estos Datasets 
cuentan con dos argumentos: transform y target_transform para modificar los ejemplos y las etiquetas.

#### Creación de modelos:
Para definir una red neuronal hay que crear una clase que herede de nn.Module. Se definen las capas de la red
en el constructor de la clase (__init__) y se especifica como se va a mandar la información en la función forward.


#### Optimización del modelo y entrenamiento:
Para entrenar un modelo se necesita una función de perdida y un optimizador.

Los entrenamientos se definen como bucles. En cada vuelta del bucle de entrenamiento se realizan predicciones del dataset 
de entrenamiento (en batches) y se propaga el error de la predicción para ajustar los parámetros del modelo.

#### Almacenar modelos:
``torch.save(model.state_dict(), "model.pth")``

#### Cargar modelos:
``model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))``

#### Inferencia:
``` 
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
``` 

### Creación de un dataset a partir de datos propios
Otro tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

En los datasets propios se deben implementar tres funciones: __init__, __len__ y __getitem__. Las imágenes se almacenan en 
un directorio y las etiquetas se almacenan separadas en un CSV.

Las transformaciones se implementan como clases para que no sea necesario pasar los parametros de transformación cada vez que se llamen.

Se pueden crear transformaciones compuestas utilizando torchvision.transforms.Compose


### TorchVision Object Detection Finetuning Tutorial
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Dataset --> getitem tiene que devolver un diccionario con la siguiente información:
* boxes (FloatTensor[N, 4])
* labels (Int64tensor[N])
* image_id (Int64Tensor[1])
* area (Tensor[N])
* iscrowd (UInt8Tensor[N])
* masks (UInt8Tensor[N,H,W]) (opcional)

Ejemplo de __getitem__:
```
def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
```