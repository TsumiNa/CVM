CVM
===

### Provides ###
combined cluster variation method with real spaces cluster expansion to 
calculated solubility limit.

## Usage ##
cvm is a python module which can run in a python3 environment.
The simplest way to run a calculation can do something like this:
```py
$ python -m cvm -inp <your input> -backend <your postprcess>
```
`-inp` is necessary for a calculation.you can see the `input.json` this dir.
`-backend` is optional arguments to do some postprcess with results. also see `backend.py` under this dir.

All the calculated results will be stored in `log` dir by the created date.