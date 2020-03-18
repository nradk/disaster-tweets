# Real or Not? Disaster Tweet Classification

In this project we classify tweets into two classes: *disaster tweets*, which
are talking about a real natural diaster and *non-disaster* tweets, which are
talking about something else.

## Environment Setup

Take the following steps to setup your environment for running training and/or
inference:

1. **Download Data**
Download the data files from the
[Kaggle Competition webpage](https://www.kaggle.com/c/nlp-getting-started/data)
and place them a directory named `data/` in the project directory. There should
be (at least) two files in `data/`: `test.csv` and `train.csv`.

2. **Setup Virtual Environment**
Create a virtual environment in the project directory if you haven't already.
This is optional but using virtual environments and frozen dependencies makes
certain things easier.

```
$ python3 -m venv .venv
```

Now activate the virtual environment:

```
$ source .venv/bin/activate
```

And install our required packages:

```
$ pip install -r requirements.txt
```

Deactivate the virtual environment if you're done working with the project

```
$ deactivate
```

3. **Download SpaCy Language Model**
We are using SpaCy to pre-process text. SpaCy needs the language model file we
will use to be downloaded beforehand. To do that, run

```
$ python3 -m spacy download en_core_web_sm
```
