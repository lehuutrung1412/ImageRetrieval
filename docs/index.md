# Content-based Image Retrieval System
Build content-based image retrieval system using deep learning, applied some large scale similarity search technicals like Kdtree, LSH, Faiss.

## Demo

## Usage

```
Usage:  python app/app.py [--large] [--feature] [--pca]
        Run demo app.
Options:
  --large=['kdtree', 'lsh', 'faiss']    Large scale method
  --feature=[PATH]                      Features indexing file path
  --pca=[INT]                           Enable pca
```

```
Usage:  python setup/export_feature.py [-path]
        Export feature indexing file to path.
Options:
  -path=[PATH]    Path to save features file
```

```
Usage:  python setup/evaluation.py [--large] [--feature] [--pca] [--top]
        Run system evaluation.
Options:
  --large=['kdtree', 'lsh', 'faiss']    Large scale method
  --feature=[PATH]                      Features indexing file path
  --pca=[INT]                           Enable pca
  --top=[INT][REQUIRED]                 Number of ranked lists element
```

## Run

### Run with Google Colab or Jupyter Notebook (Recommend with Colab resources)

1. Upload ImageRetrieval.ipynb to your Colab
2. Run all cells and go to address like xyz.ngrok.io to use

### Run with docker

#### Run from remote docker image

Pull lastest image from docker hub and run

```bash
docker pull lehuutrung1412/image-retrieval
docker run -d -p 5000:5000 lehuutrung1412/image-retrieval
```

#### Build and run from source

1. Clone sourcecode

```bash
git clone https://github.com/lehuutrung1412/ImageRetrieval.git
```

2. Build docker image and run.

```bash
docker build -t image-retrieval .
docker run -d -p 5000:5000 image-retrieval
```
