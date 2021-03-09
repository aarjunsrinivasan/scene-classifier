# scene-classifier

1. The final output of the problem is a binary classification problem. 
2. Initially I found out the cosine similarity between consecutivve shots.
3. As the four features place,act,cast,audio are distinct so I wanted to fit a nonline boundary., I chose KNN classifier.
4. Since there is impbalance in datasets I went ahead with sampling techniques and as expected Oversampling performed well compared to undersampling.


### Check install.md for package installations

### Given coarse preictions: 

```Scores: {
    "AP": 0.4418872028438688,
    "mAP": 0.4564401595678161,
    "Miou": 0.45414800530021754,
    "Precision": 0.2761656092479825,
    "Recall": 0.7473442326299846,
    "F1": 0.39309552999275693
}
```

### Baseline 1: KNN 
Train Test split 0.75/0.25
Test accuracy is :  0.9235327420267975
```
Scores: {
    "AP": 0.30079042370763087,
    "mAP": 0.3106786272526226,
    "Miou": 0.3426166826535341,
    "Precision": 0.5342522938717706,
    "Recall": 0.12604315828627943,
    "F1": 0.19920247085457896
}

```

### Baseline 2: KNN with Oversampling
Test accuracy is :  0.7355727495753915
```
Scores: {
    "AP": 0.5812168468330086,
    "mAP": 0.5735931851621817,
    "Miou": 0.4793941305957929,
    "Precision": 0.276096372587527,
    "Recall": 0.8623173522748953,
    "F1": 0.4087102162264981
}
```

### Baseline 3: KNN with Undersamplimg

Test accuracy is :  0.6485752028684657

```
Scores: {
    "AP": 0.17974113103317563,
    "mAP": 0.19112205712931624,
    "Miou": 0.37837250328419675,
    "Precision": 0.16354380841309596,
    "Recall": 0.7612370742132829,
    "F1": 0.26256056850200127
}
```

### Baseline 4: KNN with Oversampling followed by Undersamplimg

Test accuracy is :  0.6843177958105303
1```
Scores: {
    "AP": 0.2634912915127264,
    "mAP": 0.27305655507348414,
    "Miou": 0.43162796376145063,
    "Precision": 0.21485679610619718,
    "Recall": 0.8768767300525099,
    "F1": 0.33727663186040563
}
```
## Preliminary Results

The Best Performing model is KNN with oversampling with features corresponding to 
cosine similarity between adjancent features of the shots.

This Baseline 2 model is better than the given coarse prediction results

## Next trying with Random Forest Classifier with Oversampling

Test accuracy is :  0.9522537909168971
``` 
Scores: {
    "AP": 0.8231448030014414,
    "mAP": 0.8183934846996804,
    "Miou": 0.7100984502390212,
    "Precision": 0.6279891510330122,
    "Recall": 0.8546849823773823,
    "F1": 0.7161831235608485
}
```

This Performs better than KNN 

## KNN Fine Tuning
Test accuract is :  0.7470843555387809
```
Scores: {
    "AP": 0.7264627480785202,
    "mAP": 0.7146891065555759,
    "Miou": 0.6663586410503086,
    "Precision": 0.5494326852458175,
    "Recall": 0.8637291228154672,
    "F1": 0.663004900009928
}
```

Default parameters for Random forest produce good results

## Final Result

Random Forest Classifier model could fit the data well and produce non-linear decision boundaries for this scene classification task