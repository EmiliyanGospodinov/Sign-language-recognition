# Hand Gesture Recognition

In this project, we would develop a hand gesture recognition application using neural networks. It would be helpful for many areas: disability assistant, games, etc.

## Dataset

We're using the [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) dataset from Kaggle.

- The dataset format is patterned to match closely with the classic MNIST.

- Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and NO cases for `9=J` or `25=Z` because of gesture motions).

  <img src="./assets/amer_sign2.png" alt="American Sign language" style="zoom:80%;" />

- The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255.

  ![American Sign language grayscale](./assets/amer_sign3.png)

## Install dependencies

```bash
pip install -r requirements.txt
```

## Training

The implmentation of models can be found in folder `models`.

Configuration (dataset path, hyperparameters, etc) is defined in `config.yaml`.

To start training, run

```bash
python train.py
```

After training, the best model will be saved.

## Evaluation

To evaluate the trained model on the test set, run

```bash
python test.py --model <trained-model>
```

