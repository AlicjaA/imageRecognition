
### Train and save trained model
Use this command to train the model and save model
```bash
 python3 training.py -m 3_ep.hdf5 -e 3 -df data2.pkl -lf label2.pkl -l 0.025 -mn 0.4 >> Logging/3_ep.txt
```

### Test the model
- copy TEST forlder from downloaded dataset to dataset directory
- run this command
```bash
python3 test_model.py -d dataset/TEST -lf testlabels.pkl -df datatest2.pkl -lf labeltest2.pkl -m SavedModel/3_ep.hdf5 >> Logging/3_test.txt
```

