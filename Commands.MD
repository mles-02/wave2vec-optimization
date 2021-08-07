## Create Enviroment

```python
conda create -n base-library python==3.7.0
```

## Active Enviroment

```python
conda activate base-library
```

## Install Dependencies

```python
pip install tensorflow==2.5.0
```

## Export Dependencies

```python
conda list -e > requirements.txt
```