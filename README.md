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