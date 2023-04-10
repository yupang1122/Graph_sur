# Graph_Surv

Graph_sur utilizes Graph Convolutional neural network and Cox proportional hazards model to complete survive for histopathology images.

**For an updated implementation of the [Cox loss function](https://github.com/runopti/stg/blob/master/python/stg/losses.py) in PyTorch, please see [Feature Selection using Stochastic Gates (STG) by Yamada et al.](https://github.com/runopti/stg/tree/master/python).**



# Usage

### Requirements:

Download a local copy of Graph_sur and install from the directory:

	git clone https://github.com/yupang1122/Graph_sur
	cd graph_sur
	pip install .

this project is based on Python3.7 and pytorch, see more sitepackages in requirements.txt

```
./graph_sur/requiremetns.txt
```

### Prepare data

You can see more details about colon rectal(CRC) classification, stage and survive analyze data in 

```
./graph_sur/graph_sur/dataset/README.md
```

### cluster and aggregate patch feature

```
./graph_sur/graph_sur/dataset/get_cluster.py
feature,predict = Model(torch.tensor(image).resize(1,3,200,200).to(torch.float32).cuda())
```

after cluster patches you can aggregate every cluster feature in 

```
./graph_sur/graph_sur/dataset/get_cluster_feature.py
```

### Training model 

There has several pretrain mode in this project, you can see more details in 

```
./graph_sur/graphj_sur/model/models
./graph_sur/graphj_sur/dataset/models
```

And the how to use these models as follows py files

```
./graph_sur/graphj_sur/dataset/get_cluster_feature.py
./graph_sur/graphj_sur/extract_patch_feature.py
./graph_sur/graph_sur/train_sur_gcn.py
```

After install sitebackages , you can simply train survive model  in 

	./graph_sur/graphj_sur/survive_analyse/train_sur.py
	patient_risk,_ = model(patient_adjancency,patient_feature)

### Testing model

Calculate the concordance score in 

```
"./graph_sur/tests/test_sur.py"
```

to test model performence

```
if (message[patient_name]['survive_time'] >= message[target]['survive_time'] and patient_risk >= target_risk) or (message[patient_name]['survive_time'] <= message[target]['survive_time'] and patient_risk <= target_risk):    correct_number += 1
```



# Experiments results

The directory includes parts of checkpoints result, you can check and use them in 

```
./graph_sur/experimrnts
```

