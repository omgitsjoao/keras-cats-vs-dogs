## Cats Vs Dogs

This is a code from the [Keras Blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) that introduces transfer learning
using VGG16 and swapping it's top layers into a Dense and a Sigmoid output one

To run this code you will need to provide pictures from dogs and cats, you can use
the ones (and they are the one I used to model this) from [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)
competition create the folder structure like this and place the images under them

     data
        ├───train
        │   ├───dogs
        │   └───cats
        ├───dev
        │   ├───dogs
        │   └───cats
        └───test
            ├───dogs
            └───cats
            
You will also need Python 3.5 and Tensorflow, the requirements file contains versions of Tensorflow for GPU, you can install from it using
`pip install -f requeriments.txt`
