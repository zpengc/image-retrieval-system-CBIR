# image-search-system-CBIR
✍️updating



## :star:how to run this program

```python
# Windows:
pip install -r requirments.txt	# install dependencies
set FLASK_APP=start.py	# entry point
set FLASK_ENV=development	# for debug
flask extract	# offline:extract features from images
flask run		# online:interact with users via web 
flask evaluate	# output the evaluation metrics of system

#Linux:
replace 'set' with 'export' in the command above
```



## :floppy_disk:Dataset

UKBench DataSet [click here to download](https://archive.org/details/ukbench)

:sunflower: this system use first 2000 images of UKBench Dataset

![UKBench Dataset](https://raw.githubusercontent.com/zpengc/image-retrieval-system-CBIR/12866f1b47c138ed4a6eba9b462e3102417597c7/images/Examples-of-the-images-in-the-UkBench-dataset-The-datasets-consists-of-groups-of-four_W640.jpg)

This Dataset has 2550 groups each of which has 4 images and anyone of such 4 images can be input as query image. The correct result should be 4 images in the same group.

## :pushpin:Project Structure

<span style='color:blue'>pay attention to the data folder structure</span>

```python
- .idea
- app
- data
	- sift
	- ukbench
	- bof.pkl
	- inv.pkl
	- ukbench.zip
- venv
- .gitignore
- bog.py
- ...... remaining files
```

## :gift:Interface

```index page```

![](https://raw.githubusercontent.com/zpengc/image-retrieval-system-CBIR/main/images/Snipaste_2021-05-12_19-04-29.png)

```result page```

![](https://raw.githubusercontent.com/zpengc/image-retrieval-system-CBIR/main/images/Snipaste_2021-05-20_13-06-23.png)



## :bell:reference

| index | title                                                        |
| :---- | ------------------------------------------------------------ |
| 1     | Distinctive image features from scale-invariant keypoints    |
| 2     | Object Recognition from Local Scale-Invariant Features.      |
| 3     | Lost in quantization: Improving particular object retrieval in large scale image databases |
| 4     | Hamming embedding and weak geometric consistency for large scale image search |
| 5     | Hamming embedding similarity-based image classification      |
| 6     | Three things everyone should know to improve object retrieval |
| 7     | Improving bag-of-features for large scale image search       |
| 8     | On the burstiness of visual elements                         |
| 9     | Asymmetric hamming embedding: taking the best of our bits for large scale image search |
| 10    | Content-based image retrieval and feature extraction: a comprehensive review |
| 11    | Bag-of-words representation in image annotation              |

