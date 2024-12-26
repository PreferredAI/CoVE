# sbr

## Installation
```zsh
git clone --recursive git@github.com:PreferredAI/CoVE.git
cd CoVE
conda env create -f cove.yml
conda activate cove
cd cornacc
python setup.py build_ext -i
cd ..
ln -s cornacc/cornac cornac
```

## How to run
```zsh
python main.py --batch_size 32
```
This will run the `MoVE` model on the `Diginetica` dataset, with the 1x4 architecture 1x(`BPR`, `GRU4Rec`, `FPMC`, `SASRec`).

## Dataset
  * Session data (session data by default):
    * [x] Diginetica: 8k-12k lines; small
    * [x] RetailRocket: 230k lines; medium size
    * [x] Cosmetics: 2M lines; large
  * Conversion from Rating $\to$ session:
    * [ ] XING: 750k lines
    * [ ] Lastfm: 4M lines

For now we prefer to use the original session datasets.
